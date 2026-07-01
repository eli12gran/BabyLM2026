#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from itertools import chain
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers import ModernBertConfig, ModernBertForMaskedLM



os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TRAINING_CONFIG = {
    "model_name": "modernbert_base_tokmix",
    "hidden_size": 768,
    "num_hidden_layers": 22,
    "num_attention_heads": 12,
    "intermediate_size": 1152,
    "training": {
        "batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
        "mlm_probability": 0.15,
    },
    "data": {
        "max_seq_length": 256,
        "adjusted_budget_per_lang": 33_333_333,
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
    },
    "output_dir": "./model_tokmix_modernbert",
    "babylm_checkpoint_dir": "./babylm_checkpoints_tokmix_modernbert",
    "detailed_checkpoint_dir": "./checkpoints_detailed_tokmix_modernbert",
    "tokmix_tokenizer_dir": "./tokenizers_tokmix_fixed",
    "special_tokens": {
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
        "cls_token": "<CLS>",
        "sep_token": "<SEP>",
        "mask_token": "<MASK>",
    },
    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
        200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000,
        700_000_000, 800_000_000, 900_000_000, 1_000_000_000,
    ],
}

OFFICIAL_BYTE_PREMIUM = {
    "eng": 1.000000,
    "nld": 1.051606,
    "zho": 0.935966,
}

HF_DATASETS = {
    "eng": "BabyLM-community/BabyLM-2026-Strict",
    "nld": "BabyLM-community/babylm-nld",
    "zho": "BabyLM-community/babylm-zho",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModernBERT with TOKMIX shared tokenizer.")
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=TRAINING_CONFIG["tokmix_tokenizer_dir"],
        help="Directory containing TOKMIX tokenizer, usually tokenizers_tokmix_fixed/hf_tokmix_tokenizer/tokenizer.json.",
    )
    parser.add_argument("--output_dir", type=str, default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--babylm_checkpoint_dir", type=str, default=TRAINING_CONFIG["babylm_checkpoint_dir"])
    parser.add_argument("--detailed_checkpoint_dir", type=str, default=TRAINING_CONFIG["detailed_checkpoint_dir"])
    parser.add_argument("--adjusted_budget_per_lang", type=int, default=TRAINING_CONFIG["data"]["adjusted_budget_per_lang"])
    parser.add_argument("--max_examples_per_lang", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=TRAINING_CONFIG["data"]["max_seq_length"])
    parser.add_argument("--mlm_probability", type=float, default=TRAINING_CONFIG["training"]["mlm_probability"])
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def resolve_tokmix_hf_tokenizer_dir(tokenizer_dir: Optional[str] = None) -> str:
    base_candidates = []
    if tokenizer_dir is not None:
        base_candidates.append(tokenizer_dir)

    base_candidates.extend([
        TRAINING_CONFIG["tokmix_tokenizer_dir"],
        "./tokenizers_tokmix_fixed",
        "./tokenizers_tokmix",
        "./tokenizer_tokmix",
        "./tokenizer_tookmix",
        ".",
    ])

    checked = []
    for base in base_candidates:
        candidates = [os.path.join(base, "hf_tokmix_tokenizer"), base]
        for candidate in candidates:
            tokenizer_json = os.path.join(candidate, "tokenizer.json")
            checked.append(tokenizer_json)
            if os.path.exists(tokenizer_json):
                return candidate

    checked_msg = "\n".join(f"  - {p}" for p in checked)
    raise FileNotFoundError(
        "Could not find TOKMIX tokenizer.json. Checked:\n"
        f"{checked_msg}\n\n"
        "Expected example:\n"
        "  ./tokenizers_tokmix_fixed/hf_tokmix_tokenizer/tokenizer.json"
    )


def load_tokmix_tokenizer(tokenizer_dir: Optional[str] = None) -> PreTrainedTokenizerFast:
    hf_tokenizer_dir = resolve_tokmix_hf_tokenizer_dir(tokenizer_dir)
    logger.info("=" * 70)
    logger.info("Loading TOKMIX shared tokenizer")
    logger.info(f"Tokenizer directory: {hf_tokenizer_dir}")
    logger.info("=" * 70)

    tokenizer_file = os.path.join(hf_tokenizer_dir, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        **TRAINING_CONFIG["special_tokens"],
    )

    old_vocab_size = len(tokenizer)
    added = tokenizer.add_special_tokens(TRAINING_CONFIG["special_tokens"])
    new_vocab_size = len(tokenizer)
    if added > 0:
        logger.warning(
            f"Added {added} special tokens. Vocab changed from {old_vocab_size:,} to {new_vocab_size:,}. "
            "This is okay only if those specials were missing from tokenizer.json."
        )

    tokenizer.model_max_length = TRAINING_CONFIG["data"]["max_seq_length"]

    required = {
        "unk_token_id": tokenizer.unk_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "mask_token_id": tokenizer.mask_token_id,
    }
    missing = {k: v for k, v in required.items() if v is None}
    if missing:
        raise ValueError(f"TOKMIX tokenizer is missing required special-token IDs: {missing}")

    logger.info("✓ Loaded TOKMIX tokenizer")
    logger.info(f"Vocabulary size: {len(tokenizer):,}")
    logger.info(f"UNK token/id: {tokenizer.unk_token!r} / {tokenizer.unk_token_id}")
    logger.info(f"PAD token/id: {tokenizer.pad_token!r} / {tokenizer.pad_token_id}")
    logger.info(f"CLS token/id: {tokenizer.cls_token!r} / {tokenizer.cls_token_id}")
    logger.info(f"SEP token/id: {tokenizer.sep_token!r} / {tokenizer.sep_token_id}")
    logger.info(f"MASK token/id: {tokenizer.mask_token!r} / {tokenizer.mask_token_id}")
    return tokenizer


def count_official_tokens(text: str, lang: str, zho_counter_tokenizer=None) -> int:
    if not text:
        return 0
    if lang in {"eng", "nld"}:
        return len(text.split())
    if lang == "zho":
        if zho_counter_tokenizer is None:
            raise ValueError("zho_counter_tokenizer is required for Chinese official token counting.")
        return len(zho_counter_tokenizer.encode(text, add_special_tokens=False))
    raise ValueError(f"Unknown language: {lang}")


def load_training_datasets(
    adjusted_budget_per_lang: int = 33_333_333,
    max_examples_per_lang: Optional[int] = None,
) -> Tuple[Dict[str, Dataset], Dict[str, dict]]:
    logger.info("=" * 70)
    logger.info("LOADING DATASETS WITH BYTE-PREMIUM ADJUSTMENT")
    logger.info("=" * 70)
    logger.info(f"Adjusted budget per language: {adjusted_budget_per_lang:,}")
    if max_examples_per_lang is not None:
        logger.warning(f"DEBUG CAP ENABLED: max_examples_per_lang={max_examples_per_lang:,}")

    logger.info("Loading Qwen tokenizer for Chinese official token counting...")
    zho_counter_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

    datasets = {}
    budget_report = {}

    for lang, hf_path in HF_DATASETS.items():
        logger.info("-" * 70)
        logger.info(f"Loading {lang} from {hf_path}")
        raw_dataset = load_dataset(hf_path, split="train", trust_remote_code=True)

        byte_premium = OFFICIAL_BYTE_PREMIUM[lang]
        official_tokens = 0
        adjusted_tokens = 0.0
        selected_indices = []

        for idx, example in enumerate(raw_dataset):
            if max_examples_per_lang is not None and len(selected_indices) >= max_examples_per_lang:
                break

            text = example.get("text", "")
            n_tokens = count_official_tokens(text, lang, zho_counter_tokenizer=zho_counter_tokenizer)
            if n_tokens <= 0:
                continue

            official_tokens += n_tokens
            adjusted_tokens = official_tokens * byte_premium
            selected_indices.append(idx)

            if adjusted_tokens >= adjusted_budget_per_lang:
                break

        if not selected_indices:
            raise RuntimeError(f"No examples selected for {lang}.")

        datasets[lang] = raw_dataset.select(selected_indices)
        budget_report[lang] = {
            "examples": len(selected_indices),
            "official_tokens": official_tokens,
            "official_byte_premium": byte_premium,
            "adjusted_tokens": adjusted_tokens,
        }

        logger.info(
            f"✓ {lang}: selected {len(selected_indices):,} examples | "
            f"official_tokens={official_tokens:,} | adjusted_tokens={adjusted_tokens:,.0f} | "
            f"byte_premium={byte_premium}"
        )

    total_adjusted = sum(x["adjusted_tokens"] for x in budget_report.values())
    logger.info("=" * 70)
    logger.info("DATA BUDGET REPORT")
    logger.info(json.dumps(budget_report, indent=2))
    logger.info(f"Total adjusted budget: {total_adjusted:,.0f}")
    logger.info("✓ Within 100M adjusted BabyLM budget" if total_adjusted <= 100_000_000 else "✗ Over 100M adjusted BabyLM budget")
    logger.info("=" * 70)

    return datasets, budget_report


def preprocess_language_dataset_for_mlm(
    dataset: Dataset,
    lang: str,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
) -> Dataset:
    logger.info("-" * 70)
    logger.info(f"Preprocessing {lang} with shared TOKMIX tokenizer for MLM")

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    content_len = max_seq_length - 2

    if cls_id is None or sep_id is None:
        raise ValueError("TOKMIX tokenizer must define cls_token_id and sep_token_id.")
    if content_len <= 0:
        raise ValueError("max_seq_length must be at least 3.")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return {"input_ids": tokenized["input_ids"]}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {lang} with TOKMIX",
    )

    def chunk_function(examples):
        all_input_ids = list(chain.from_iterable(examples["input_ids"]))
        total_length = (len(all_input_ids) // content_len) * content_len
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}

        content_chunks = [
            all_input_ids[i: i + content_len]
            for i in range(0, total_length, content_len)
        ]
        sequences = [[cls_id] + chunk + [sep_id] for chunk in content_chunks]
        return {
            "input_ids": sequences,
            "attention_mask": [[1] * max_seq_length for _ in sequences],
        }

    chunked_dataset = tokenized_dataset.map(
        chunk_function,
        batched=True,
        batch_size=chunk_batch_size,
        remove_columns=tokenized_dataset.column_names,
        desc=f"Chunking {lang} for MLM",
    )

    logger.info(f"✓ {lang}: {len(chunked_dataset):,} fixed MLM sequences of {max_seq_length} TOKMIX IDs")
    return chunked_dataset


def preprocess_tokmix_datasets_for_mlm(
    datasets: Dict[str, Dataset],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
    seed: int = 42,
) -> Dataset:
    logger.info("=" * 70)
    logger.info("PREPROCESSING DATASETS: SHARED TOKMIX + MODERNBERT MLM")
    logger.info("=" * 70)

    chunked_by_lang = []
    for lang, dataset in datasets.items():
        chunked_by_lang.append(
            preprocess_language_dataset_for_mlm(
                dataset=dataset,
                lang=lang,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                tokenize_batch_size=tokenize_batch_size,
                chunk_batch_size=chunk_batch_size,
            )
        )

    train_dataset = concatenate_datasets(chunked_by_lang)
    train_dataset = train_dataset.shuffle(seed=seed)

    logger.info("=" * 70)
    logger.info(f"Combined train dataset: {len(train_dataset):,} sequences")
    logger.info(f"Sequence length: {max_seq_length}")
    logger.info(f"Approx BPE positions per epoch including CLS/SEP: {len(train_dataset) * max_seq_length:,}")
    logger.info("=" * 70)
    return train_dataset


class TokenCounterCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerFast],
        adjusted_words_per_epoch: float,
        train_dataset_size: int,
        max_seq_length: int = 256,
        checkpoint_intervals=None,
        output_dir: str = "./babylm_checkpoints_tokmix_modernbert",
    ):
        self.tokenizer = tokenizer
        self.adjusted_words_per_epoch = float(adjusted_words_per_epoch)
        self.train_dataset_size = int(train_dataset_size)
        self.max_seq_length = int(max_seq_length)
        self.checkpoint_intervals = checkpoint_intervals or []
        self.output_dir = output_dir
        self.total_words_seen = 0.0
        self.checkpoints_saved = set()
        os.makedirs(self.output_dir, exist_ok=True)

    def _adjusted_words_per_optimizer_step(self, args) -> float:
        sequences_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        return (sequences_per_step / self.train_dataset_size) * self.adjusted_words_per_epoch

    def _bpe_tokens_per_optimizer_step(self, args) -> int:
        return args.per_device_train_batch_size * args.gradient_accumulation_steps * self.max_seq_length

    def _checkpoint_name(self, checkpoint_words: int) -> str:
        if checkpoint_words < 1_000_000:
            return f"chck_{checkpoint_words // 1_000}K"
        return f"chck_{checkpoint_words // 1_000_000}M"

    def _save_tokenizer(self, save_dir: str):
        if self.tokenizer is None:
            logger.warning("No tokenizer object provided; tokenizer not saved.")
            return []
        self.tokenizer.save_pretrained(save_dir)
        return sorted([
            name for name in os.listdir(save_dir)
            if name.startswith("tokenizer") or name in {"special_tokens_map.json", "tokenizer_config.json"}
        ])

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_words: int) -> None:
        checkpoint_name = self._checkpoint_name(checkpoint_words)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer_assets = self._save_tokenizer(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}
        approx_bpe_seen = state.global_step * self._bpe_tokens_per_optimizer_step(args)

        metadata = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_words": checkpoint_words,
            "checkpoint_words_millions": checkpoint_words / 1_000_000,
            "actual_words_seen_estimate": self.total_words_seen,
            "adjusted_words_per_epoch": self.adjusted_words_per_epoch,
            "train_dataset_size_sequences": self.train_dataset_size,
            "bpe_tokens_per_optimizer_step": self._bpe_tokens_per_optimizer_step(args),
            "approx_bpe_tokens_seen": approx_bpe_seen,
            "global_step": state.global_step,
            "epoch": safe_float(state.epoch),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate")),
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "timestamp": datetime.now().isoformat(),
            "tokenizer_type": "TOKMIX shared tokenizer",
            "tokenizer_assets_saved": tokenizer_assets,
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "tokenization_strategy": "TOKMIX",
            "model_architecture": "ModernBERT",
            "objective": "masked_language_modeling",
            "exposure_unit": "byte_premium_adjusted_whitespace_words",
        }

        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"Saved BabyLM-style evaluation checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Requested word threshold: {checkpoint_words:,} adjusted words")
        logger.info(f"Actual estimated exposure: {self.total_words_seen:,.0f} adjusted words")
        logger.info(f"Approx BPE positions seen: {approx_bpe_seen:,}")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"TOKMIX tokenizer assets saved: {tokenizer_assets}")
        logger.info("=" * 70)

    def on_train_begin(self, args, state, control, **kwargs):
        words_per_step = self._adjusted_words_per_optimizer_step(args)
        self.total_words_seen = state.global_step * words_per_step
        self.checkpoints_saved = {
            checkpoint_words
            for checkpoint_words in self.checkpoint_intervals
            if checkpoint_words <= self.total_words_seen
        }

        logger.info("=" * 70)
        logger.info("BABYLM-STYLE WORD-EXPOSURE CHECKPOINT CALLBACK INITIALIZED")
        logger.info(f"Checkpoint output dir: {self.output_dir}")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Adjusted words per epoch: {self.adjusted_words_per_epoch:,.0f}")
        logger.info(f"Train dataset size (sequences): {self.train_dataset_size:,}")
        logger.info(f"Adjusted words per optimizer step: {words_per_step:,.1f}")
        logger.info(f"BPE positions per optimizer step: {self._bpe_tokens_per_optimizer_step(args):,}")
        logger.info(f"Initial estimated adjusted words seen: {self.total_words_seen:,.0f}")
        logger.info(f"Already passed checkpoint thresholds: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            logger.warning("No model found in callback kwargs; cannot save checkpoint.")
            return control

        words_per_step = self._adjusted_words_per_optimizer_step(args)
        self.total_words_seen = state.global_step * words_per_step

        for checkpoint_words in self.checkpoint_intervals:
            if self.total_words_seen >= checkpoint_words and checkpoint_words not in self.checkpoints_saved:
                self.checkpoints_saved.add(checkpoint_words)
                self._save_babylm_checkpoint(model=model, args=args, state=state, checkpoint_words=checkpoint_words)

        return control


class DetailedCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: str, save_every_n_steps: int = 1000):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info = {}
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"DetailedCheckpointCallback: JSON logs every {save_every_n_steps} steps")

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if current_step % self.save_every_n_steps == 0 and current_step > 0:
            self._save_checkpoint_info(current_step, state, args)

    def _save_checkpoint_info(self, step: int, state, args) -> None:
        latest_log = state.log_history[-1] if state.log_history else {}
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate", args.learning_rate)),
            "epoch": safe_float(state.epoch),
            "total_steps": state.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }
        progress_pct = (step / state.max_steps * 100) if state.max_steps and state.max_steps > 0 else 0.0
        checkpoint_data["progress_percent"] = progress_pct

        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json")
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)

        self.checkpoint_info[step] = checkpoint_data
        master_log_file = os.path.join(self.checkpoint_dir, "checkpoint_log.json")
        with open(master_log_file, "w", encoding="utf-8") as f:
            json.dump(self.checkpoint_info, f, indent=2)

        loss_display = checkpoint_data["loss"]
        loss_text = f"{loss_display:.4f}" if loss_display is not None else "NA"
        epoch_text = f"{checkpoint_data['epoch']:.2f}" if checkpoint_data["epoch"] is not None else "NA"

        logger.info(f"Detailed JSON checkpoint {step}: Loss={loss_text} | Epoch={epoch_text} | Progress={progress_pct:.1f}%")


def create_model(tokenizer: PreTrainedTokenizerFast, max_seq_length: int) -> ModernBertForMaskedLM:
    logger.info("Creating ModernBERT masked language model with TOKMIX vocabulary")

    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=TRAINING_CONFIG["hidden_size"],
        num_hidden_layers=TRAINING_CONFIG["num_hidden_layers"],
        num_attention_heads=TRAINING_CONFIG["num_attention_heads"],
        intermediate_size=TRAINING_CONFIG["intermediate_size"],
        max_position_embeddings=max_seq_length,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        tie_word_embeddings=True,
    )

    model = ModernBertForMaskedLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created ModernBERT MLM with {num_params:,} parameters")
    logger.info(f"TOKMIX vocab size: {len(tokenizer):,}")
    logger.info(f"Max sequence length: {max_seq_length}")
    return model


def setup_training(
    model: ModernBertForMaskedLM,
    train_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    adjusted_words_per_epoch: float,
    max_seq_length: int,
    mlm_probability: float,
) -> Tuple[Trainer, TokenCounterCallback]:
    logger.info("Setting up ModernBERT TOKMIX MLM training")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["training"]["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["training"]["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["training"]["gradient_accumulation_steps"],
        logging_steps=TRAINING_CONFIG["training"]["logging_steps"],
        save_strategy="steps",
        save_steps=500,
        learning_rate=TRAINING_CONFIG["training"]["learning_rate"],
        warmup_steps=TRAINING_CONFIG["training"]["warmup_steps"],
        weight_decay=TRAINING_CONFIG["training"]["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["training"]["max_grad_norm"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim="adamw_torch",
        gradient_checkpointing=True,
        seed=TRAINING_CONFIG["training"]["seed"],
        save_total_limit=5,
        remove_unused_columns=False,
        report_to=[],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    token_callback = TokenCounterCallback(
        tokenizer=tokenizer,
        adjusted_words_per_epoch=adjusted_words_per_epoch,
        train_dataset_size=len(train_dataset),
        max_seq_length=max_seq_length,
        checkpoint_intervals=TRAINING_CONFIG["checkpoint_intervals"],
        output_dir=TRAINING_CONFIG["babylm_checkpoint_dir"],
    )

    detailed_callback = DetailedCheckpointCallback(
        checkpoint_dir=TRAINING_CONFIG["detailed_checkpoint_dir"],
        save_every_n_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[token_callback, detailed_callback],
    )

    bpe_tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max_seq_length
    )

    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Architecture: ModernBERT-base-like")
    logger.info("Tokenizer strategy: TOKMIX shared tokenizer")
    logger.info("Objective: Masked Language Modeling")
    logger.info(f"TOKMIX vocab size: {len(tokenizer):,}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Total epochs: {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"MLM probability: {mlm_probability}")
    logger.info(f"Total sequences: {len(train_dataset):,}")
    logger.info(f"Sequence length: {max_seq_length}")
    logger.info(f"BPE positions per optimizer step: {bpe_tokens_per_step:,}")
    logger.info(f"Adjusted words per epoch: {adjusted_words_per_epoch:,.0f}")
    logger.info(f"Approx BPE positions per epoch: {len(train_dataset) * max_seq_length:,}")
    logger.info(f"Normal Trainer checkpoint dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM-style checkpoint dir: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info(f"Detailed JSON checkpoint dir: {TRAINING_CONFIG['detailed_checkpoint_dir']}")
    logger.info("=" * 70)

    return trainer, token_callback


def copy_tokmix_metadata(tokenizer_dir: str, final_dir: str) -> list:
    copied = []
    source_root = tokenizer_dir
    if os.path.basename(os.path.normpath(source_root)) == "hf_tokmix_tokenizer":
        possible_roots = [source_root, os.path.dirname(source_root)]
    else:
        possible_roots = [source_root, os.path.join(source_root, "hf_tokmix_tokenizer")]

    optional_names = [
        "metadata.json",
        "intrinsic_metrics.json",
        "tokmix_vocab.json",
        "tokmix_tokenizer.json",
    ]

    for root in possible_roots:
        if not os.path.isdir(root):
            continue
        for name in optional_names:
            src = os.path.join(root, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(final_dir, name))
                copied.append(name)

    return sorted(set(copied))


def save_final_bundle(
    model: ModernBertForMaskedLM,
    tokenizer: PreTrainedTokenizerFast,
    tokenizer_dir: str,
    final_dir: str,
    budget_report: Dict[str, dict],
) -> None:
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    copied_metadata = copy_tokmix_metadata(tokenizer_dir, final_dir)

    with open(os.path.join(final_dir, "final_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model_architecture": "ModernBERT",
                "architecture_size": "base-like",
                "objective": "masked_language_modeling",
                "tokenization_strategy": "TOKMIX",
                "tokenizer_type": "shared_tokmix_tokenizer",
                "vocab_size": len(tokenizer),
                "tokenizer_metadata_copied": copied_metadata,
                "budget_report": budget_report,
                "training_config": TRAINING_CONFIG,
            },
            f,
            indent=2,
        )

    logger.info(f"✓ Final ModernBERT TOKMIX bundle saved to {final_dir}")
    logger.info(f"✓ TOKMIX metadata copied: {copied_metadata}")


def main() -> None:
    args = parse_args()

    TRAINING_CONFIG["tokmix_tokenizer_dir"] = args.tokenizer_dir
    TRAINING_CONFIG["output_dir"] = args.output_dir
    TRAINING_CONFIG["babylm_checkpoint_dir"] = args.babylm_checkpoint_dir
    TRAINING_CONFIG["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    TRAINING_CONFIG["data"]["adjusted_budget_per_lang"] = args.adjusted_budget_per_lang
    TRAINING_CONFIG["data"]["max_seq_length"] = args.max_seq_length
    TRAINING_CONFIG["training"]["mlm_probability"] = args.mlm_probability

    seed = TRAINING_CONFIG["training"]["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("=" * 70)
    logger.info("BabyLM 2026 Multilingual Track - ModernBERT + TOKMIX")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"TOKMIX tokenizer dir: {args.tokenizer_dir}")
    logger.info(f"Output dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 70)

    logger.info("[1/5] Loading shared TOKMIX tokenizer")
    tokenizer = load_tokmix_tokenizer(args.tokenizer_dir)

    logger.info("[2/5] Loading same selected datasets as GPT-2/TOKMIX scripts")
    datasets, budget_report = load_training_datasets(
        adjusted_budget_per_lang=args.adjusted_budget_per_lang,
        max_examples_per_lang=args.max_examples_per_lang,
    )

    adjusted_words_per_epoch = sum(x["adjusted_tokens"] for x in budget_report.values())

    logger.info("[3/5] Tokenizing/chunking datasets for ModernBERT MLM")
    train_dataset = preprocess_tokmix_datasets_for_mlm(
        datasets=datasets,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=TRAINING_CONFIG["data"]["tokenize_batch_size"],
        chunk_batch_size=TRAINING_CONFIG["data"]["chunk_batch_size"],
        seed=seed,
    )

    logger.info("[4/5] Creating ModernBERT MLM model")
    model = create_model(tokenizer, max_seq_length=args.max_seq_length)

    logger.info("[5/5] Setting up Trainer")
    trainer, token_callback = setup_training(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        adjusted_words_per_epoch=adjusted_words_per_epoch,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
    )

    logger.info("=" * 70)
    logger.info("STARTING MODERNBERT TOKMIX MLM TRAINING")
    logger.info("=" * 70)
    logger.info(
        f"Selected adjusted corpus budget: {args.adjusted_budget_per_lang:,} per language "
        f"({args.adjusted_budget_per_lang * 3:,} target total adjusted tokens)"
    )
    logger.info(f"Actual adjusted words per epoch: {adjusted_words_per_epoch:,.0f}")
    logger.info("Data selection matches GPT-2/TOKMIX scripts if the same dataset revisions are resolved.")
    logger.info("Seed=42 is used for sequence shuffle and Trainer RNG.")
    logger.info(f"Normal Trainer checkpoints: {TRAINING_CONFIG['output_dir']}/checkpoint-*")
    logger.info(f"BabyLM-style checkpoints: {TRAINING_CONFIG['babylm_checkpoint_dir']}/chck_*")
    logger.info("=" * 70)

    train_kwargs = {}
    if args.resume_from_checkpoint:
        train_kwargs["resume_from_checkpoint"] = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    trainer.train(**train_kwargs)

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(
        f"Exposure checkpoints saved: "
        f"{len(token_callback.checkpoints_saved)}/{len(TRAINING_CONFIG['checkpoint_intervals'])}"
    )
    logger.info(f"Total estimated adjusted-word exposure: {token_callback.total_words_seen:,.0f}")
    logger.info("=" * 70)

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    save_final_bundle(
        model=model,
        tokenizer=tokenizer,
        tokenizer_dir=args.tokenizer_dir,
        final_dir=final_dir,
        budget_report=budget_report,
    )

    checkpoint_dir = TRAINING_CONFIG["output_dir"]
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")])
        logger.info(f"✓ Saved {len(checkpoints)} HF Trainer recovery checkpoints in {checkpoint_dir}")
        logger.info(f"Recovery checkpoints: {checkpoints}")

    babylm_dir = TRAINING_CONFIG["babylm_checkpoint_dir"]
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted([d for d in os.listdir(babylm_dir) if d.startswith("chck_")])
        logger.info(f"✓ Saved {len(babylm_checkpoints)} BabyLM-style checkpoints in {babylm_dir}")
        logger.info(f"BabyLM-style checkpoints: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
