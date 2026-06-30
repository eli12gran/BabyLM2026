import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime
from itertools import chain
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

try:
    from transformers import ModernBertConfig, ModernBertForMaskedLM
except ImportError as exc:
    raise ImportError(
        "ModernBERT is not available in your installed transformers version. "
        "Install or upgrade transformers, for example:\n\n"
        "  python -m pip install -U transformers accelerate tokenizers datasets safetensors\n\n"
        "Then rerun this script."
    ) from exc

from tokenizer_utilities_nooverlap import NOOVERLAPTokenizer, SPECIAL_TOKENS


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

TRAINING_CONFIG = {
    "model_name": "modernbert",

    # ModernBERT-base-like defaults:
    # hidden_size=768, num_attention_heads=12, num_hidden_layers=22,
    # intermediate_size=1152.
    #
    # This is a different architecture from GPT-2, so these are not meant to
    # match GPT-2 layer-for-layer. The DATA selection remains matched.
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
        # Same target used in your GPT-2 NOOVERLAP/TOKMIX scripts:
        # 100M adjusted BabyLM units / 3 languages.
        "adjusted_budget_per_lang": 33_333_333,
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
    },

    "output_dir": "./model_nooverlap_modernbert",
    "babylm_checkpoint_dir": "./babylm_checkpoints_nooverlap_modernbert",
    "detailed_checkpoint_dir": "./checkpoints_detailed_nooverlap_modernbert",

    # Same BabyLM exposure checkpoint schedule as your GPT-2 scripts.
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


LANGUAGE_TAGS = {
    "eng": "<EN>",
    "nld": "<NL>",
    "zho": "<ZH>",
}


DEFAULT_TOKENIZER_DIR = "./tokenizers_nooverlap_fixed"
LANGUAGES = ("eng", "nld", "zho")


# ============================================================================
# ARGS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ModernBERT with correct NOOVERLAP tokenization."
    )

    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default=DEFAULT_TOKENIZER_DIR,
        help=(
            "Directory containing nested NOOVERLAP tokenizers, e.g. "
            "tokenizer_eng/tokenizer_eng.json, tokenizer_nld/tokenizer_nld.json, "
            "and tokenizer_zho/tokenizer_zho.json."
        ),
    )
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
        help="Optional debug cap. Leave unset for official/comparable training.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional HF Trainer checkpoint path, e.g. ./model_nooverlap_modernbert/checkpoint-16000.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=TRAINING_CONFIG["data"]["max_seq_length"],
        help="Training sequence length. Default 256 to match the GPT-2 runs.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=TRAINING_CONFIG["training"]["mlm_probability"],
        help="MLM masking probability. Default 0.15.",
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
# TOKENIZER LOADING
# ============================================================================

def get_monolingual_tokenizer_path(tokenizer_dir: str, lang: str) -> str:
    tokenizer_path = os.path.join(
        tokenizer_dir,
        f"tokenizer_{lang}",
        f"tokenizer_{lang}.json",
    )

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found for {lang}: {tokenizer_path}")

    return tokenizer_path


def load_tokenizer_with_unk(path: str) -> Tokenizer:
    """
    Load a tokenizers.Tokenizer JSON and ensure Unigram has unk_id set.

    This prevents:
        Encountered an unknown token but `unk_id` is missing

    It does not retrain or change the learned vocabulary; it only sets the local
    unknown-token id to the existing <UNK> entry if missing.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})

    if model.get("type") == "Unigram" and model.get("unk_id") is None:
        vocab = model.get("vocab", [])

        unk_id = None
        for idx, item in enumerate(vocab):
            if isinstance(item, list) and len(item) >= 1 and item[0] == "<UNK>":
                unk_id = idx
                break

        if unk_id is None:
            raise ValueError(
                f"Could not set unk_id for {path}: <UNK> was not found in model.vocab"
            )

        model["unk_id"] = int(unk_id)
        data["model"] = model

    return Tokenizer.from_str(json.dumps(data, ensure_ascii=False))


def load_monolingual_tokenizers(tokenizer_dir: str = DEFAULT_TOKENIZER_DIR) -> Dict[str, Tokenizer]:
    logger.info("Loading existing monolingual Unigram tokenizers")

    tokenizers = {}

    for lang in LANGUAGES:
        tokenizer_path = get_monolingual_tokenizer_path(tokenizer_dir, lang)
        tokenizers[lang] = load_tokenizer_with_unk(tokenizer_path)
        vocab_size = len(tokenizers[lang].get_vocab())

        logger.info(f"✓ Loaded {lang}: {vocab_size:,} local tokens from {tokenizer_path}")

    return tokenizers


def create_nooverlap_tokenizer(tokenizers: Dict[str, Tokenizer]) -> NOOVERLAPTokenizer:
    """
    Create the same NOOVERLAP combiner:
      - shared special-token block
      - disjoint language-specific vocabulary segments
      - local_id -> global_id maps per language
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
# DATASET LOADING: SAME SELECTION LOGIC AS GPT-2 RUNS
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
    Same data selection logic as the GPT-2 NOOVERLAP script:
      - same HF datasets
      - same byte-premium factors
      - same adjusted budget per language
      - deterministic first-N selection until the budget threshold is reached

    There is no random sampling here. The seed affects the later shuffle only.
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

        budget_report[lang] = {
            "examples": len(selected_indices),
            "official_tokens": official_tokens,
            "official_byte_premium": byte_premium,
            "adjusted_tokens": adjusted_tokens,
        }

        logger.info(
            f"✓ {lang}: selected {len(selected_indices):,} examples | "
            f"official_tokens={official_tokens:,} | "
            f"adjusted_tokens={adjusted_tokens:,.0f} | "
            f"byte_premium={byte_premium}"
        )

    total_adjusted = sum(x["adjusted_tokens"] for x in budget_report.values())
    logger.info("=" * 70)
    logger.info("DATA BUDGET REPORT")
    logger.info(json.dumps(budget_report, indent=2))
    logger.info(f"Total adjusted budget: {total_adjusted:,.0f}")
    logger.info("=" * 70)

    return datasets


# ============================================================================
# MODERNBERT / MLM PREPROCESSING
# ============================================================================

def preprocess_language_dataset_for_mlm(
    dataset: Dataset,
    lang: str,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
) -> Dataset:
    """
    Tokenize one language correctly for ModernBERT MLM:
      1. Use the monolingual tokenizer for `lang`.
      2. Remap local monolingual IDs to global NOOVERLAP IDs.
      3. Concatenate all IDs for that language.
      4. Chunk into max_seq_length - 2 content tokens.
      5. Wrap every chunk as [CLS] + content + [SEP].

    Labels are NOT created here. The MLM collator creates labels dynamically.
    """
    logger.info("-" * 70)
    logger.info(f"Preprocessing {lang} with correct NOOVERLAP remapping for MLM")

    cls_id = nooverlap_tokenizer.cls_token_id
    sep_id = nooverlap_tokenizer.sep_token_id
    content_len = max_seq_length - 2

    if content_len <= 0:
        raise ValueError("max_seq_length must be at least 3 for [CLS] content [SEP].")

    def tokenize_function(examples):
        texts = examples["text"]
        input_ids = nooverlap_tokenizer.encode_batch(
            texts,
            language=lang,
            add_special_tokens=False,
        )
        return {"input_ids": input_ids}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=tokenize_batch_size,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing/remapping {lang}",
    )

    def chunk_function(examples):
        all_input_ids = list(chain.from_iterable(examples["input_ids"]))

        total_length = (len(all_input_ids) // content_len) * content_len
        if total_length == 0:
            return {
                "input_ids": [],
                "attention_mask": [],
            }

        content_chunks = [
            all_input_ids[i: i + content_len]
            for i in range(0, total_length, content_len)
        ]

        sequences = [
            [cls_id] + chunk + [sep_id]
            for chunk in content_chunks
        ]

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

    logger.info(
        f"✓ {lang}: {len(chunked_dataset):,} fixed MLM sequences of {max_seq_length} global NOOVERLAP IDs"
    )
    return chunked_dataset


def preprocess_nooverlap_datasets_for_mlm(
    datasets: Dict[str, Dataset],
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    max_seq_length: int = 256,
    tokenize_batch_size: int = 1000,
    chunk_batch_size: int = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Preprocess each language separately, then concatenate and shuffle.

    This matches the GPT-2 script's fairness properties:
      - same language-specific tokenization
      - same global NOOVERLAP remapping
      - same deterministic seed for the final sequence shuffle
    """
    logger.info("=" * 70)
    logger.info("PREPROCESSING DATASETS: CORRECT NOOVERLAP + MODERNBERT MLM")
    logger.info("=" * 70)

    chunked_by_lang = []
    for lang, dataset in datasets.items():
        chunked_by_lang.append(
            preprocess_language_dataset_for_mlm(
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
    logger.info(f"Approx corpus tokens per epoch including CLS/SEP: {len(train_dataset) * max_seq_length:,}")
    logger.info("=" * 70)

    return train_dataset


# ============================================================================
# MLM COLLATOR
# ============================================================================

class NOOVERLAPMLMCollator:
    """
    Dynamic MLM collator for global NOOVERLAP IDs.

    It implements standard BERT-style masking:
      - choose 15% of non-special, non-padding positions
      - labels are original token IDs at masked positions
      - labels are -100 elsewhere
      - of selected positions:
          80% -> <MASK>
          10% -> random token
          10% -> original token

    Special tokens are never masked:
      <UNK>, <PAD>, <CLS>, <SEP>, <MASK>
    """

    def __init__(
        self,
        pad_token_id: int,
        mask_token_id: int,
        special_token_ids: List[int],
        vocab_size: int,
        mlm_probability: float = 0.15,
        seed: int = 42,
    ):
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.special_token_ids = set(int(x) for x in special_token_ids)
        self.vocab_size = int(vocab_size)
        self.mlm_probability = float(mlm_probability)

        # The Trainer/DataLoader will call this collator repeatedly.
        # torch seed is also set globally via set_seed; this generator makes
        # the masking stream reproducible enough for a fixed worker setup.
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item.get("attention_mask", [1] * len(item["input_ids"])) for item in batch]

        max_len = max(len(x) for x in input_ids)

        padded_input_ids = []
        padded_attention_mask = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * padding_len)
            padded_attention_mask.append(mask + [0] * padding_len)

        inputs = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability, dtype=torch.float)

        # Do not mask padding or special tokens.
        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for special_id in self.special_token_ids:
            special_mask |= labels.eq(special_id)

        special_mask |= attention_mask_tensor.eq(0)
        probability_matrix.masked_fill_(special_mask, value=0.0)

        masked_indices = torch.bernoulli(
            probability_matrix,
            generator=self.generator,
        ).bool()

        labels[~masked_indices] = -100

        # 80% replace with [MASK]
        replace_prob = torch.full(labels.shape, 0.8, dtype=torch.float)
        indices_replaced = torch.bernoulli(
            replace_prob,
            generator=self.generator,
        ).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% replace with random token. This is 50% of the remaining 20%.
        random_prob = torch.full(labels.shape, 0.5, dtype=torch.float)
        indices_random = (
            torch.bernoulli(random_prob, generator=self.generator).bool()
            & masked_indices
            & ~indices_replaced
        )

        random_words = torch.randint(
            low=0,
            high=self.vocab_size,
            size=labels.shape,
            dtype=torch.long,
            generator=self.generator,
        )
        inputs[indices_random] = random_words[indices_random]

        # Remaining 10% keep original.

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
        }


# ============================================================================
# CALLBACKS
# ============================================================================

class TokenCounterCallback(TrainerCallback):
    """
    Save BabyLM-style exposure checkpoints at token-exposure intervals.

    Note:
      For MLM, these are exposure checkpoints, not causal next-token checkpoints.
      The exposure counter mirrors your GPT-2 scripts:
        global_step * batch_size * grad_accum * max_seq_length
    """

    def __init__(
        self,
        nooverlap_tokenizer: NOOVERLAPTokenizer,
        max_seq_length: int = 256,
        checkpoint_intervals=None,
        output_dir: str = "./babylm_checkpoints_nooverlap_modernbert",
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
        copied = []

        for lang in LANGUAGES:
            src = get_monolingual_tokenizer_path(self.tokenizer_asset_dir, lang)
            dst_name = f"tokenizer_{lang}.json"
            dst = os.path.join(save_dir, dst_name)

            shutil.copy2(src, dst)
            copied.append(dst_name)

        # Copy root metadata and optional wrappers if available.
        optional_assets = [
            "metadata.json",
            "tokenizer_utilities_nooverlap.py",
            "tokenization_nooverlap.py",
        ]

        for name in optional_assets:
            src = os.path.join(self.tokenizer_asset_dir, name)
            dst = os.path.join(save_dir, name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(name)

        return copied

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_tokens: int) -> None:
        checkpoint_name = self._checkpoint_name(checkpoint_tokens)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)
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
            "model_architecture": "ModernBERT",
            "objective": "masked_language_modeling",
            "note": (
                "ModernBERT MLM checkpoint trained on global NOOVERLAP IDs. "
                "Monolingual local IDs were remapped to disjoint language-specific global segments."
            ),
        }

        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"Saved BabyLM-style evaluation checkpoint: {checkpoint_name}")
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
        logger.info("BABYLM-STYLE EXPOSURE CHECKPOINT CALLBACK INITIALIZED")
        logger.info(f"Checkpoint output dir: {self.output_dir}")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
        logger.info(f"Initial estimated tokens seen: {self.total_tokens_seen:,}")
        logger.info(f"Already passed checkpoint thresholds: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            logger.warning("No model found in callback kwargs; cannot save checkpoint.")
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
    """

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

        logger.info(
            f"Detailed JSON checkpoint {step}: Loss={loss_text} | "
            f"Epoch={epoch_text} | Progress={progress_pct:.1f}%"
        )


# ============================================================================
# MODEL
# ============================================================================

def create_model(nooverlap_tokenizer: NOOVERLAPTokenizer, max_seq_length: int) -> ModernBertForMaskedLM:
    logger.info("Creating ModernBERT masked language model")

    config = ModernBertConfig(
        vocab_size=nooverlap_tokenizer.get_vocab_size(),
        hidden_size=TRAINING_CONFIG["hidden_size"],
        num_hidden_layers=TRAINING_CONFIG["num_hidden_layers"],
        num_attention_heads=TRAINING_CONFIG["num_attention_heads"],
        intermediate_size=TRAINING_CONFIG["intermediate_size"],
        max_position_embeddings=max_seq_length,
        pad_token_id=nooverlap_tokenizer.pad_token_id,
        bos_token_id=nooverlap_tokenizer.cls_token_id,
        eos_token_id=nooverlap_tokenizer.sep_token_id,
        cls_token_id=nooverlap_tokenizer.cls_token_id,
        sep_token_id=nooverlap_tokenizer.sep_token_id,
        # ModernBertConfig may not define mask_token_id explicitly in all versions,
        # but PretrainedConfig accepts extra kwargs and stores them.
        mask_token_id=nooverlap_tokenizer.mask_token_id,
        tie_word_embeddings=True,
    )

    model = ModernBertForMaskedLM(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created ModernBERT MLM with {num_params:,} parameters")
    logger.info(f"Model vocab size: {config.vocab_size:,}")
    logger.info(f"Max sequence length: {max_seq_length}")
    logger.info(
        "Special IDs: "
        f"UNK={nooverlap_tokenizer.unk_token_id}, "
        f"PAD={nooverlap_tokenizer.pad_token_id}, "
        f"CLS={nooverlap_tokenizer.cls_token_id}, "
        f"SEP={nooverlap_tokenizer.sep_token_id}, "
        f"MASK={nooverlap_tokenizer.mask_token_id}"
    )

    return model


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(
    model: ModernBertForMaskedLM,
    train_dataset: Dataset,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    tokenizer_asset_dir: str,
    max_seq_length: int,
    mlm_probability: float,
) -> tuple:
    logger.info("Setting up ModernBERT MLM training")

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

    collator = NOOVERLAPMLMCollator(
        pad_token_id=nooverlap_tokenizer.pad_token_id,
        mask_token_id=nooverlap_tokenizer.mask_token_id,
        special_token_ids=[
            nooverlap_tokenizer.unk_token_id,
            nooverlap_tokenizer.pad_token_id,
            nooverlap_tokenizer.cls_token_id,
            nooverlap_tokenizer.sep_token_id,
            nooverlap_tokenizer.mask_token_id,
        ],
        vocab_size=nooverlap_tokenizer.get_vocab_size(),
        mlm_probability=mlm_probability,
        seed=TRAINING_CONFIG["training"]["seed"],
    )

    token_callback = TokenCounterCallback(
        nooverlap_tokenizer=nooverlap_tokenizer,
        max_seq_length=max_seq_length,
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
        data_collator=collator,
        callbacks=[token_callback, detailed_callback],
    )

    tokens_per_step = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max_seq_length
    )

    logger.info("=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info("Architecture: ModernBERT")
    logger.info("Objective: Masked Language Modeling")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Total epochs: {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"MLM probability: {mlm_probability}")
    logger.info(f"Total sequences: {len(train_dataset):,}")
    logger.info(f"Sequence length: {max_seq_length}")
    logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
    logger.info(f"Approx tokens per epoch: {len(train_dataset) * max_seq_length:,}")
    logger.info(f"Exposure checkpoints: {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"Normal Trainer checkpoint dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM-style checkpoint dir: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info(f"Detailed JSON checkpoint dir: {TRAINING_CONFIG['detailed_checkpoint_dir']}")
    logger.info("=" * 70)

    return trainer, token_callback


# ============================================================================
# SAVE FINAL BUNDLE
# ============================================================================

def save_final_bundle(
    model: ModernBertForMaskedLM,
    nooverlap_tokenizer: NOOVERLAPTokenizer,
    tokenizer_asset_dir: str,
    final_dir: str,
) -> None:
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    nooverlap_tokenizer.save(final_dir)

    copied = []

    for lang in LANGUAGES:
        src = get_monolingual_tokenizer_path(tokenizer_asset_dir, lang)
        dst_name = f"tokenizer_{lang}.json"

        shutil.copy2(src, os.path.join(final_dir, dst_name))
        copied.append(dst_name)

    for name in [
        "metadata.json",
        "tokenizer_utilities_nooverlap.py",
        "tokenization_nooverlap.py",
    ]:
        src = os.path.join(tokenizer_asset_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(final_dir, name))
            copied.append(name)

    # Add BERT-style special-token names for downstream loading.
    special_tokens_map_path = os.path.join(final_dir, "special_tokens_map.json")
    with open(special_tokens_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "unk_token": nooverlap_tokenizer.unk_token,
                "pad_token": nooverlap_tokenizer.pad_token,
                "bos_token": nooverlap_tokenizer.cls_token,
                "eos_token": nooverlap_tokenizer.sep_token,
                "cls_token": nooverlap_tokenizer.cls_token,
                "sep_token": nooverlap_tokenizer.sep_token,
                "mask_token": nooverlap_tokenizer.mask_token,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save a tokenizer_config that documents the required language routing.
    tokenizer_config_path = os.path.join(final_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "NOOVERLAPHFTokenizer",
                "auto_map": {
                    "AutoTokenizer": [
                        "tokenization_nooverlap.NOOVERLAPHFTokenizer",
                        None,
                    ]
                },
                "trust_remote_code": True,
                "strategy": "NOOVERLAP",
                "requires_language_for_encoding": True,
                "default_language": "eng",
                "special_tokens": nooverlap_tokenizer.special_tokens,
                "languages": nooverlap_tokenizer.languages,
                "language_tags": nooverlap_tokenizer.language_tags,
                "unk_token": nooverlap_tokenizer.unk_token,
                "pad_token": nooverlap_tokenizer.pad_token,
                "bos_token": nooverlap_tokenizer.cls_token,
                "eos_token": nooverlap_tokenizer.sep_token,
                "cls_token": nooverlap_tokenizer.cls_token,
                "sep_token": nooverlap_tokenizer.sep_token,
                "mask_token": nooverlap_tokenizer.mask_token,
                "unk_token_id": nooverlap_tokenizer.unk_token_id,
                "pad_token_id": nooverlap_tokenizer.pad_token_id,
                "bos_token_id": nooverlap_tokenizer.cls_token_id,
                "eos_token_id": nooverlap_tokenizer.sep_token_id,
                "cls_token_id": nooverlap_tokenizer.cls_token_id,
                "sep_token_id": nooverlap_tokenizer.sep_token_id,
                "mask_token_id": nooverlap_tokenizer.mask_token_id,
                "model_max_length": model.config.max_position_embeddings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(final_dir, "final_model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model_architecture": "ModernBERT",
                "objective": "masked_language_modeling",
                "tokenization_strategy": "NOOVERLAP",
                "vocab_size": nooverlap_tokenizer.get_vocab_size(),
                "language_id_ranges": nooverlap_tokenizer.language_id_ranges,
                "tokenizer_assets_copied": copied,
                "training_config": TRAINING_CONFIG,
                "note": (
                    "For valid NOOVERLAP evaluation, set tokenizer_config.json "
                    "default_language to eng/nld/zho according to the evaluation language."
                ),
            },
            f,
            indent=2,
        )

    logger.info(f"✓ Final ModernBERT NOOVERLAP bundle saved to {final_dir}")
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
    logger.info("BabyLM 2026 Multilingual Track - ModernBERT + CORRECT NOOVERLAP")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Tokenizer dir: {args.tokenizer_dir}")
    logger.info(f"Output dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"Seed: {seed}")
    logger.info("=" * 70)

    logger.info("[1/5] Loading existing monolingual tokenizers")
    monolingual_tokenizers = load_monolingual_tokenizers(args.tokenizer_dir)
    nooverlap_tokenizer = create_nooverlap_tokenizer(monolingual_tokenizers)

    root_tokenizer_dir = os.path.join(TRAINING_CONFIG["output_dir"], "nooverlap_tokenizer_assets")
    os.makedirs(root_tokenizer_dir, exist_ok=True)
    nooverlap_tokenizer.save(root_tokenizer_dir)

    logger.info("[2/5] Loading same selected datasets as GPT-2 scripts")
    datasets = load_training_datasets(
        adjusted_budget_per_lang=args.adjusted_budget_per_lang,
        max_examples_per_lang=args.max_examples_per_lang,
    )

    logger.info("[3/5] Tokenizing/remapping/chunking datasets for MLM")
    train_dataset = preprocess_nooverlap_datasets_for_mlm(
        datasets=datasets,
        nooverlap_tokenizer=nooverlap_tokenizer,
        max_seq_length=args.max_seq_length,
        tokenize_batch_size=TRAINING_CONFIG["data"]["tokenize_batch_size"],
        chunk_batch_size=TRAINING_CONFIG["data"]["chunk_batch_size"],
        seed=seed,
    )

    logger.info("[4/5] Creating ModernBERT MLM model")
    model = create_model(nooverlap_tokenizer, max_seq_length=args.max_seq_length)

    logger.info("[5/5] Setting up Trainer")
    trainer, token_callback = setup_training(
        model=model,
        train_dataset=train_dataset,
        nooverlap_tokenizer=nooverlap_tokenizer,
        tokenizer_asset_dir=args.tokenizer_dir,
        max_seq_length=args.max_seq_length,
        mlm_probability=args.mlm_probability,
    )

    logger.info("=" * 70)
    logger.info("STARTING MODERNBERT MLM TRAINING")
    logger.info("=" * 70)
    logger.info(
        f"Selected adjusted corpus budget: {args.adjusted_budget_per_lang:,} per language "
        f"({args.adjusted_budget_per_lang * 3:,} total adjusted tokens)"
    )
    logger.info("Data selection matches GPT-2 scripts if the same dataset revisions are resolved.")
    logger.info("The seed=42 is used for sequence shuffle and MLM masking RNG.")
    logger.info(f"Normal Trainer checkpoints: {TRAINING_CONFIG['output_dir']}/checkpoint-*")
    logger.info(f"BabyLM-style exposure checkpoints: {TRAINING_CONFIG['babylm_checkpoint_dir']}/chck_*")
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
        logger.info(f"✓ Saved {len(babylm_checkpoints)} exposure checkpoints in {babylm_dir}")
        logger.info(f"Exposure checkpoints: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
