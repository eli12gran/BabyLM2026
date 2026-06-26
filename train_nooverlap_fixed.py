import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from tokenizer_utilities_nooverlap import NOOVERLAPTokenizer, SPECIAL_TOKENS


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TRAINING_CONFIG = {
    "model_name": "gpt2",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,

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
    },

    "data": {
        "max_seq_length": 256,
        "adjusted_budget_per_lang": 33_333_333,
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
    },

    "output_dir": "./model_nooverlap",
    "babylm_checkpoint_dir": "./babylm_checkpoints",
    "detailed_checkpoint_dir": "./checkpoints_detailed",

    # BabyLM 2026 Strict-style exposure checkpoints.
    # These are saved separately from HF Trainer recovery checkpoints.
    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
        200_000_000, 300_000_000, 400_000_000, 500_000_000, 600_000_000,
        700_000_000, 800_000_000, 900_000_000, 1_000_000_000,
    ],
}


# BabyLM byte-premium factors used by your previous multilingual scripts.
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


LANGUAGE_TAGS = {
    "eng": "<EN>",
    "nld": "<NL>",
    "zho": "<ZH>",
}

DEFAULT_TOKENIZER_DIR = "./tokenizers_nooverlap_fixed"
LANGUAGES = ("eng", "nld", "zho")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 with correct NOOVERLAP tokenization.")

    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_TOKENIZER_DIR, help="Directory containing tokenizer_eng/nld/zho.json.")
    parser.add_argument("--output_dir", type=str, default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--babylm_checkpoint_dir", type=str, default=TRAINING_CONFIG["babylm_checkpoint_dir"])
    parser.add_argument("--detailed_checkpoint_dir", type=str, default=TRAINING_CONFIG["detailed_checkpoint_dir"])

    parser.add_argument(
        "--adjusted_budget_per_lang",
        type=int,
        default=TRAINING_CONFIG["data"]["adjusted_budget_per_lang"],
        help="Byte-premium adjusted token budget per language. Default is 100M/3.",
    )
    parser.add_argument(
        "--max_examples_per_lang",
        type=int,
        default=None,
        help="Optional debug cap. Leave unset for official training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional HF Trainer checkpoint path, e.g. ./model_nooverlap/checkpoint-16000.",
    )

    return parser.parse_args()


def safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


# ============================================================================
# TOKENIZER LOADING AND CORRECT NOOVERLAP COMBINATION
# ============================================================================

def load_monolingual_tokenizers(tokenizer_dir: str = DEFAULT_TOKENIZER_DIR) -> Dict[str, Tokenizer]:
    logger.info("Loading monolingual Unigram tokenizers")

    tokenizers = {}

    for lang in LANGUAGES:
        tokenizer_path = os.path.join(
            tokenizer_dir,
            f"tokenizer_{lang}",
            f"tokenizer_{lang}.json",
        )

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found for {lang}: {tokenizer_path}"
            )

        tokenizers[lang] = Tokenizer.from_file(tokenizer_path)
        vocab_size = len(tokenizers[lang].get_vocab())

        logger.info(f"✓ Loaded {lang}: {vocab_size:,} local tokens from {tokenizer_path}")

    return tokenizers


def create_nooverlap_tokenizer(tokenizers: Dict[str, Tokenizer]) -> NOOVERLAPTokenizer:
    """
    Create the correct NOOVERLAP combiner.

    The resulting object contains:
      - one shared special-token block
      - one disjoint token-ID segment per language
      - local_id -> global_id maps for each language
    """
    nooverlap_tokenizer = NOOVERLAPTokenizer(
        tokenizers_dict=tokenizers,
        language_tags=LANGUAGE_TAGS,
        special_tokens=SPECIAL_TOKENS,
    )

    logger.info("=" * 70)
    logger.info("NOOVERLAP TOKENIZER")
    logger.info("=" * 70)
    logger.info(f"Total global vocabulary size: {nooverlap_tokenizer.get_vocab_size():,}")
    logger.info(f"Shared special tokens: {SPECIAL_TOKENS}")
    logger.info("Per-language disjoint segments:")
    for lang, id_range in nooverlap_tokenizer.language_id_ranges.items():
        logger.info(
            f"  {lang}: [{id_range['start_inclusive']:,}, {id_range['end_exclusive']:,}) "
            f"size={id_range['size_excluding_shared_specials']:,}"
        )

    sizes = nooverlap_tokenizer.get_vocab_size_per_language(include_shared_specials=False)
    if len(set(sizes.values())) != 1:
        logger.warning(
            "NOOVERLAP expects the same non-special vocabulary size per language. "
            f"Current sizes: {sizes}. This may be okay only if your monolingual tokenizers "
            "were intentionally trained with slightly different realized sizes."
        )
    else:
        logger.info(f"✓ Equal non-special vocab size per language: {next(iter(sizes.values())):,}")

    logger.info("=" * 70)
    return nooverlap_tokenizer


# ============================================================================
# DATASET LOADING WITH BYTE-PREMIUM ADJUSTMENT
# ============================================================================

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
) -> Dict[str, Dataset]:
    """
    Load the multilingual training corpora and select a byte-premium-adjusted
    budget per language.

    This keeps the earlier cheap-training setup:
      total adjusted budget ~= 3 * 33,333,333 = 100M adjusted tokens.
    With num_epochs=10, total exposure can reach ~= 1B training-token exposures.
    """
    logger.info("=" * 70)
    logger.info("LOADING DATASETS WITH BYTE-PREMIUM ADJUSTMENT")
    logger.info("=" * 70)
    logger.info(f"Adjusted budget per language: {adjusted_budget_per_lang:,}")
    if max_examples_per_lang is not None:
        logger.warning(f"DEBUG CAP ENABLED: max_examples_per_lang={max_examples_per_lang:,}")

    logger.info("Loading Qwen tokenizer for Chinese official token counting...")
    zho_counter_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )

    datasets = {}

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
            n_tokens = count_official_tokens(
                text,
                lang,
                zho_counter_tokenizer=zho_counter_tokenizer,
            )

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

        logger.info(
            f"✓ {lang}: selected {len(selected_indices):,} examples | "
            f"official_tokens={official_tokens:,} | "
            f"adjusted_tokens={adjusted_tokens:,.0f} | "
            f"byte_premium={byte_premium}"
        )

    logger.info("=" * 70)
    return datasets


# ============================================================================
# CORRECT NOOVERLAP PREPROCESSING
# ============================================================================

def preprocess_language_dataset(
    dataset: Dataset,
    lang: str,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
) -> Dataset:
    """
    Tokenize one language correctly:
      1. Use the monolingual tokenizer for `lang`.
      2. Remap local monolingual IDs to global NOOVERLAP IDs.
      3. Chunk into fixed-length CLM sequences.
    """
    logger.info("-" * 70)
    logger.info(f"Preprocessing {lang} with correct NOOVERLAP remapping")

    def tokenize_function(examples):
        texts = examples["text"]
        input_ids = nooverlap_tokenizer.encode_batch(
            texts,
            language=lang,
            add_special_tokens=False,
        )
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing/remapping {lang}",
    )

    def chunk_function(examples):
        # Avoid Python's sum(list_of_lists, []), which is very slow for large data.
        all_input_ids = list(chain.from_iterable(examples["input_ids"]))

        total_length = (len(all_input_ids) // max_seq_length) * max_seq_length
        if total_length == 0:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        chunks = [
            all_input_ids[i: i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]

        return {
            "input_ids": chunks,
            "attention_mask": [[1] * max_seq_length for _ in chunks],
            "labels": [chunk.copy() for chunk in chunks],
        }

    chunked_dataset = tokenized_dataset.map(
        chunk_function,
        batched=True,
        batch_size=chunk_batch_size,
        remove_columns=tokenized_dataset.column_names,
        desc=f"Chunking {lang}",
    )

    logger.info(
        f"✓ {lang}: {len(chunked_dataset):,} fixed sequences of {max_seq_length} global NOOVERLAP IDs"
    )
    return chunked_dataset


def preprocess_nooverlap_datasets(
    datasets: Dict[str, Dataset],
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Preprocess each language separately so every token is remapped into its
    language-specific NOOVERLAP segment, then concatenate and shuffle sequences.
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING DATASETS: CORRECT NOOVERLAP")
    logger.info("=" * 70)

    chunked_by_lang = []
    for lang, dataset in datasets.items():
        chunked_by_lang.append(
            preprocess_language_dataset(
                dataset=dataset,
                lang=lang,
                nooverlap_tokenizer=nooverlap_tokenizer,
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
    logger.info(f"Approx corpus tokens per epoch: {len(train_dataset) * max_seq_length:,}")
    logger.info("=" * 70)

    return train_dataset


# ============================================================================
# CUSTOM CALLBACKS FOR CHECKPOINTING
# ============================================================================

class TokenCounterCallback(TrainerCallback):
    """
    Save BabyLM evaluation checkpoints at exposure intervals while keeping
    normal Hugging Face Trainer recovery checkpoints unchanged.

    Normal recovery checkpoints:
        ./model_nooverlap/checkpoint-500
        ./model_nooverlap/checkpoint-1000
        ...

    BabyLM evaluation checkpoints:
        ./babylm_checkpoints/chck_1M
        ./babylm_checkpoints/chck_2M
        ...

    Exposure counter:
        global_step * per_device_train_batch_size * gradient_accumulation_steps * max_seq_length
    """

    def __init__(
        self,
        nooverlap_tokenizer: NOOVERLAPTokenizer,
        max_seq_length: int = 256,
        checkpoint_intervals=None,
        output_dir: str = "./babylm_checkpoints",
        tokenizer_asset_dir: str = ".",
    ):
        self.nooverlap_tokenizer = nooverlap_tokenizer
        self.max_seq_length = max_seq_length
        self.checkpoint_intervals = checkpoint_intervals or []
        self.output_dir = output_dir
        self.tokenizer_asset_dir = tokenizer_asset_dir

        self.total_tokens_seen = 0
        self.checkpoints_saved = set()

        os.makedirs(self.output_dir, exist_ok=True)

    def _tokens_per_optimizer_step(self, args) -> int:
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.max_seq_length
        )

    def _checkpoint_name(self, checkpoint_tokens: int) -> str:
        return f"chck_{checkpoint_tokens // 1_000_000}M"

    def _copy_optional_tokenizer_assets(self, save_dir: str):
        """
        Copy files needed for reconstruction/evaluation.
        tokenization_nooverlap.py is optional; it is copied if present.
        """
        asset_names = [
            "tokenizer_eng.json",
            "tokenizer_nld.json",
            "tokenizer_zho.json",
            "tokenizer_utilities.py",
            "tokenization_nooverlap.py",
        ]

        copied = []
        for name in asset_names:
            src = os.path.join(self.tokenizer_asset_dir, name)
            dst = os.path.join(save_dir, name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(name)
            elif name != "tokenization_nooverlap.py":
                logger.warning(f"Tokenizer asset not found, not copied: {src}")

        return copied

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_tokens: int) -> None:
        checkpoint_name = self._checkpoint_name(checkpoint_tokens)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)

        # Save the corrected NOOVERLAP metadata and mapping description.
        self.nooverlap_tokenizer.save(save_dir)

        copied_assets = self._copy_optional_tokenizer_assets(save_dir)

        with open(os.path.join(save_dir, "training_args.json"), "w", encoding="utf-8") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}
        metadata = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_tokens": checkpoint_tokens,
            "checkpoint_millions": checkpoint_tokens / 1_000_000,
            "actual_tokens_seen_estimate": self.total_tokens_seen,
            "global_step": state.global_step,
            "epoch": safe_float(state.epoch),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate")),
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "tokens_per_optimizer_step": self._tokens_per_optimizer_step(args),
            "timestamp": datetime.now().isoformat(),
            "tokenizer_assets_copied": copied_assets,
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "tokenization_strategy": "NOOVERLAP",
            "note": (
                "Model was trained on global NOOVERLAP IDs. Monolingual local IDs were "
                "remapped to disjoint language-specific global vocabulary segments."
            ),
        }

        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"Saved BabyLM evaluation checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Requested exposure threshold: {checkpoint_tokens:,} tokens")
        logger.info(f"Actual estimated exposure: {self.total_tokens_seen:,} tokens")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Tokenizer assets copied: {copied_assets}")
        logger.info("=" * 70)

    def on_train_begin(self, args, state, control, **kwargs):
        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step

        self.checkpoints_saved = {
            checkpoint_tokens
            for checkpoint_tokens in self.checkpoint_intervals
            if checkpoint_tokens <= self.total_tokens_seen
        }

        logger.info("=" * 70)
        logger.info("BABYLM EVALUATION CHECKPOINT CALLBACK INITIALIZED")
        logger.info(f"BabyLM checkpoint output dir: {self.output_dir}")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
        logger.info(f"Initial estimated tokens seen: {self.total_tokens_seen:,}")
        logger.info(f"Already passed BabyLM checkpoint thresholds: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            logger.warning("No model found in callback kwargs; cannot save BabyLM checkpoint.")
            return control

        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step

        for checkpoint_tokens in self.checkpoint_intervals:
            if (
                self.total_tokens_seen >= checkpoint_tokens
                and checkpoint_tokens not in self.checkpoints_saved
            ):
                self.checkpoints_saved.add(checkpoint_tokens)
                self._save_babylm_checkpoint(
                    model=model,
                    args=args,
                    state=state,
                    checkpoint_tokens=checkpoint_tokens,
                )

        return control


class DetailedCheckpointCallback(TrainerCallback):
    """
    Save lightweight JSON metadata every N optimizer steps.

    These JSON files are not Trainer checkpoints. Real resumable Trainer
    checkpoints are still saved under output_dir/checkpoint-* by TrainingArguments.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints_detailed", save_every_n_steps: int = 1000):
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

        logger.info(
            f"Detailed JSON checkpoint {step}: Loss={loss_text} | "
            f"Epoch={epoch_text} | Progress={progress_pct:.1f}%"
        )


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(nooverlap_tokenizer: NOOVERLAPTokenizer) -> GPT2LMHeadModel:
    logger.info("Creating GPT-2 model")

    config = GPT2Config(
        vocab_size=nooverlap_tokenizer.get_vocab_size(),
        n_embd=TRAINING_CONFIG["hidden_size"],
        n_layer=TRAINING_CONFIG["num_hidden_layers"],
        n_head=TRAINING_CONFIG["num_attention_heads"],
        n_inner=TRAINING_CONFIG["intermediate_size"],
        n_positions=TRAINING_CONFIG["data"]["max_seq_length"],
        n_ctx=TRAINING_CONFIG["data"]["max_seq_length"],
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=nooverlap_tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    model.config.use_cache = False  # Required/recommended with gradient checkpointing.

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model with {num_params:,} parameters")
    logger.info(f"Model vocab size: {config.vocab_size:,}")

    return model


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(
    model: GPT2LMHeadModel,
    train_dataset: Dataset,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    tokenizer_asset_dir: str,
) -> tuple:
    logger.info("Setting up training")

    # Keep memory use down on Ampere/Ada while preserving your hyperparameters.
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
        save_safetensors=True,
    )

    pad_token_id = nooverlap_tokenizer.pad_token_id

    def custom_collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_len = max(len(x) for x in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            padding_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * padding_len)
            padded_attention_mask.append(mask + [0] * padding_len)
            padded_labels.append(label + [-100] * padding_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    token_callback = TokenCounterCallback(
        nooverlap_tokenizer=nooverlap_tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
        checkpoint_intervals=TRAINING_CONFIG["checkpoint_intervals"],
        output_dir=TRAINING_CONFIG["babylm_checkpoint_dir"],
        tokenizer_asset_dir=tokenizer_asset_dir,
    )

    detailed_callback = DetailedCheckpointCallback(
        checkpoint_dir=TRAINING_CONFIG["detailed_checkpoint_dir"],
        save_every_n_steps=1000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=[token_callback, detailed_callback],
    )

    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * TRAINING_CONFIG["data"]["max_seq_length"]
    )

    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Total epochs: {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"Total sequences: {len(train_dataset):,}")
    logger.info(f"Sequence length: {TRAINING_CONFIG['data']['max_seq_length']}")
    logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
    logger.info(f"Approx tokens per epoch: {len(train_dataset) * TRAINING_CONFIG['data']['max_seq_length']:,}")
    logger.info(f"BabyLM competition checkpoints: {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info("Normal Trainer recovery checkpoints: every 500 optimizer steps")
    logger.info(f"Normal Trainer checkpoint dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM evaluation checkpoint dir: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info(f"Detailed JSON checkpoint dir: {TRAINING_CONFIG['detailed_checkpoint_dir']}")
    logger.info("=" * 70)

    return trainer, token_callback


def save_final_bundle(
    model: GPT2LMHeadModel,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    tokenizer_asset_dir: str,
    final_dir: str,
) -> None:
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    nooverlap_tokenizer.save(final_dir)

    copied = []
    for name in [
        "tokenizer_eng.json",
        "tokenizer_nld.json",
        "tokenizer_zho.json",
        "tokenizer_utilities.py",
        "tokenization_nooverlap.py",
    ]:
        src = os.path.join(tokenizer_asset_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(final_dir, name))
            copied.append(name)

    with open(os.path.join(final_dir, "final_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "tokenization_strategy": "NOOVERLAP",
                "vocab_size": nooverlap_tokenizer.get_vocab_size(),
                "language_id_ranges": nooverlap_tokenizer.language_id_ranges,
                "tokenizer_assets_copied": copied,
                "training_config": TRAINING_CONFIG,
            },
            f,
            indent=2,
        )

    logger.info(f"✓ Final model bundle saved to {final_dir}")
    logger.info(f"✓ Tokenizer assets copied: {copied}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_args()

    TRAINING_CONFIG["output_dir"] = args.output_dir
    TRAINING_CONFIG["babylm_checkpoint_dir"] = args.babylm_checkpoint_dir
    TRAINING_CONFIG["detailed_checkpoint_dir"] = args.detailed_checkpoint_dir
    TRAINING_CONFIG["data"]["adjusted_budget_per_lang"] = args.adjusted_budget_per_lang

    set_seed(TRAINING_CONFIG["training"]["seed"])

    logger.info("=" * 70)
    logger.info("BabyLM 2026 Multilingual Track - CORRECT NOOVERLAP Training")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Tokenizer dir: {args.tokenizer_dir}")
    logger.info(f"Output dir: {TRAINING_CONFIG['output_dir']}")
    logger.info("=" * 70)

    logger.info("[1/5] Loading monolingual tokenizers")
    monolingual_tokenizers = load_monolingual_tokenizers(args.tokenizer_dir)
    nooverlap_tokenizer = create_nooverlap_tokenizer(monolingual_tokenizers)

    # Save tokenizer metadata at the root output dir immediately.
    root_tokenizer_dir = os.path.join(TRAINING_CONFIG["output_dir"], "nooverlap_tokenizer_assets")
    os.makedirs(root_tokenizer_dir, exist_ok=True)
    nooverlap_tokenizer.save(root_tokenizer_dir)

    logger.info("[2/5] Loading datasets")
    datasets = load_training_datasets(
        adjusted_budget_per_lang=args.adjusted_budget_per_lang,
        max_examples_per_lang=args.max_examples_per_lang,
    )

    logger.info("[3/5] Tokenizing/remapping/chunking datasets")
    train_dataset = preprocess_nooverlap_datasets(
        datasets=datasets,
        nooverlap_tokenizer=nooverlap_tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
        tokenize_batch_size=TRAINING_CONFIG["data"]["tokenize_batch_size"],
        chunk_batch_size=TRAINING_CONFIG["data"]["chunk_batch_size"],
        seed=TRAINING_CONFIG["training"]["seed"],
    )

    logger.info("[4/5] Creating model")
    model = create_model(nooverlap_tokenizer)

    logger.info("[5/5] Setting up trainer")
    trainer, token_callback = setup_training(
        model=model,
        train_dataset=train_dataset,
        nooverlap_tokenizer=nooverlap_tokenizer,
        tokenizer_asset_dir=args.tokenizer_dir,
    )

    logger.info("=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info(
        f"Selected adjusted corpus budget: {args.adjusted_budget_per_lang:,} per language "
        f"({args.adjusted_budget_per_lang * 3:,} total adjusted tokens)"
    )
    logger.info(
        "Because num_epochs=10, exposure checkpoints can continue up to ~1B tokens "
        "depending on the number of 256-token chunks produced."
    )
    logger.info(f"Normal Trainer checkpoints: {TRAINING_CONFIG['output_dir']}/checkpoint-*")
    logger.info(f"BabyLM evaluation checkpoints: {TRAINING_CONFIG['babylm_checkpoint_dir']}/chck_*")
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
        f"BabyLM evaluation checkpoints saved: "
        f"{len(token_callback.checkpoints_saved)}/{len(TRAINING_CONFIG['checkpoint_intervals'])}"
    )
    logger.info(f"Total estimated training-token exposure: {token_callback.total_tokens_seen:,}")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    save_final_bundle(
        model=model,
        nooverlap_tokenizer=nooverlap_tokenizer,
        tokenizer_asset_dir=args.tokenizer_dir,
        final_dir=final_dir,
    )

    checkpoint_dir = TRAINING_CONFIG["output_dir"]
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")])
        logger.info(f"✓ Saved {len(checkpoints)} HF Trainer recovery checkpoints in {checkpoint_dir}")
        logger.info(f"Recovery checkpoints: {checkpoints}")

    babylm_dir = TRAINING_CONFIG["babylm_checkpoint_dir"]
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted([d for d in os.listdir(babylm_dir) if d.startswith("chck_")])
        logger.info(f"✓ Saved {len(babylm_checkpoints)} BabyLM evaluation checkpoints in {babylm_dir}")
        logger.info(f"BabyLM evaluation checkpoints: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
