import os
import shutil
import json
import logging
from typing import Dict, List, Optional
import numpy as np

import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback, AutoTokenizer
)
from tokenizers import Tokenizer
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TRAINING_CONFIG = {
    "model_name": "gpt2",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,

    "training": {
        "batch_size": 16,  # G4 can handle 8-16, use smaller to be safe
        "gradient_accumulation_steps": 8,  # Effective batch = 32
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
    },

    "data": {
        "max_seq_length": 256,  # BabyLM competition requirement
    },

    "output_dir": "./model_nooverlap",
    "babylm_checkpoint_dir": "./babylm_checkpoints",

    # BabyLM 2026 Strict competition checkpoint intervals (exposure-based)
    # These are saved separately from normal Hugging Face recovery checkpoints.
    # Strict-small requires 1M..10M and then 20M..100M.

    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000,
        60_000_000, 70_000_000, 80_000_000, 90_000_000, 100_000_000,
    ]
}

logger.info(f"BabyLM 2026 Strict Track - {len(TRAINING_CONFIG['checkpoint_intervals'])} competition checkpoints configured")


# ============================================================================
# TOKENIZER LOADING
# ============================================================================

def load_monolingual_tokenizers(tokenizer_dir: str = ".") -> Dict[str, Tokenizer]:
    """Load monolingual tokenizers for each language"""
    logger.info("Loading Monolingual Tokenizers")

    tokenizers = {}
    languages = ["eng", "nld", "zho"]

    for lang in languages:
        tokenizer_path = os.path.join(tokenizer_dir, f"tokenizer_{lang}.json")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        tokenizers[lang] = Tokenizer.from_file(tokenizer_path)
        vocab_size = len(tokenizers[lang].get_vocab())
        logger.info(f"✓ Loaded {lang}: {vocab_size} tokens")

    return tokenizers


def create_nooverlap_tokenizer(tokenizers: Dict[str, Tokenizer]) -> 'NOOVERLAPTokenizer':
    """Create NOOVERLAP tokenizer from language-specific tokenizers"""
    from tokenizer_utilities import NOOVERLAPTokenizer
    
    language_tags = {
        "eng": "<EN>",
        "nld": "<NL>",
        "zho": "<ZH>",
    }

    nooverlap_tokenizer = NOOVERLAPTokenizer(tokenizers, language_tags)

    logger.info(f"Created NOOVERLAP Tokenizer")
    logger.info(f"Total vocabulary size: {nooverlap_tokenizer.get_vocab_size()}")

    vocab_per_lang = nooverlap_tokenizer.get_vocab_size_per_language()
    for lang, size in vocab_per_lang.items():
        logger.info(f"  {lang.upper()}: {size}")

    return nooverlap_tokenizer


# ============================================================================
# DATASET LOADING WITH BYTE-PREMIUM ADJUSTMENT (BabyLM Official)
# ============================================================================

OFFICIAL_BYTE_PREMIUM = {
    "eng": 1.000000,
    "nld": 1.051606,
    "zho": 0.935966,
}


def count_official_tokens(text: str, lang: str, zho_tokenizer=None) -> int:
    """Count official tokens according to BabyLM rules"""
    if not text:
        return 0
    if lang in {"eng", "nld"}:
        return len(text.split())
    if lang == "zho":
        return len(zho_tokenizer.encode(text, add_special_tokens=False))
    raise ValueError(f"Unknown language: {lang}")


def load_training_datasets(adjusted_budget_per_lang: int = 33_333_333) -> Dict[str, Dataset]:
    """
    Load training datasets with byte-premium adjustment.
    
    Each language gets equal representation according to BabyLM byte-premium rules.
    adjusted_budget_per_lang: Target adjusted tokens per language (default: 100M / 3)
    """
    logger.info(f"Loading Training Datasets (byte-premium adjusted, budget={adjusted_budget_per_lang:,} per lang)")

    hf_datasets = {
        "eng": "BabyLM-community/BabyLM-2026-Strict",
        "nld": "BabyLM-community/babylm-nld",
        "zho": "BabyLM-community/babylm-zho",
    }

    # Load Chinese tokenizer for official token counting
    zho_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    datasets = {}

    for lang, hf_path in hf_datasets.items():
        logger.info(f"Loading {lang} with byte-premium adjustment...")
        dataset = load_dataset(hf_path, split='train', trust_remote_code=True)

        # Select examples up to the byte-premium adjusted budget
        bpf = OFFICIAL_BYTE_PREMIUM[lang]
        official_tokens = 0
        adjusted_budget = 0.0
        selected_examples = []

        for example in dataset:
            text = example["text"]
            n_tokens = count_official_tokens(text, lang, zho_tokenizer=zho_tokenizer)

            if n_tokens == 0:
                continue

            official_tokens += n_tokens
            adjusted_budget = official_tokens * bpf
            selected_examples.append(example)

            if adjusted_budget >= adjusted_budget_per_lang:
                logger.info(f"  Stopped at {official_tokens:,} official tokens, "
                           f"{adjusted_budget:,.0f} adjusted budget, "
                           f"from {len(selected_examples):,} examples")
                break

        # Create dataset from selected examples
        datasets[lang] = Dataset.from_dict({"text": [ex["text"] for ex in selected_examples]})
        logger.info(f"✓ Loaded {lang}: {len(datasets[lang]):,} examples ({adjusted_budget:,.0f} adjusted tokens)")

    return datasets


# ============================================================================
# DATA PREPARATION AND PREPROCESSING
# ============================================================================

def prepare_tagged_dataset(datasets: Dict[str, Dataset], language_tags: Dict[str, str],
                          max_examples_per_lang: Optional[int] = None) -> Dataset:
    """Prepare training data with language tags for balanced representation"""
    logger.info("Preparing tagged training data")

    combined_texts = []

    for lang, dataset in datasets.items():
        tag = language_tags[lang]
        num_samples = len(dataset) if max_examples_per_lang is None else min(max_examples_per_lang, len(dataset))

        logger.info(f"Adding {num_samples:,} examples from {lang}")
        for i, example in enumerate(dataset):
            if max_examples_per_lang and i >= max_examples_per_lang:
                break

            text = example['text']
            tagged_text = f"{tag} {text}"
            combined_texts.append({"text": tagged_text})

    logger.info(f"Created combined dataset with {len(combined_texts):,} examples")
    combined_dataset = Dataset.from_dict({"text": [d["text"] for d in combined_texts]})

    return combined_dataset


def preprocess_dataset(dataset: Dataset, tokenizer, max_seq_length: int = 256) -> Dataset:
    """
    Tokenize and preprocess dataset.
    
    Process:
    1. Tokenize all texts
    2. Chunk into sequences of max_seq_length
    """
    logger.info(f"Preprocessing dataset with {len(dataset):,} examples...")

    def tokenize_function(examples):
        tokenized = tokenizer.encode_batch(examples["text"])
        input_ids = [t.ids for t in tokenized]
        attention_mask = [t.attention_mask if hasattr(t, 'attention_mask') else [1]*len(t.ids) for t in tokenized]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    # Tokenize
    logger.info("Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    logger.info("Chunking sequences...")

    def chunk_function(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Chunk into sequences of max_seq_length
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()

        return result

    chunked_dataset = tokenized_dataset.map(
        chunk_function,
        batched=True,
        batch_size=100,
        desc="Chunking",
    )

    logger.info(f"Preprocessed dataset: {len(chunked_dataset):,} sequences of {max_seq_length} tokens")

    return chunked_dataset


# ============================================================================
# CUSTOM CALLBACKS FOR CHECKPOINTING
# ============================================================================

class TokenCounterCallback(TrainerCallback):
    """
    Save BabyLM evaluation checkpoints at exposure intervals while keeping the
    normal Hugging Face Trainer recovery checkpoints unchanged.

    Normal recovery checkpoints still happen through TrainingArguments:
        ./model_nooverlap/checkpoint-500
        ./model_nooverlap/checkpoint-1000
        ...

    BabyLM evaluation checkpoints are additionally saved here:
        ./babylm_checkpoints/chck_1M
        ./babylm_checkpoints/chck_2M
        ...
        ./babylm_checkpoints/chck_100M

    The exposure counter is based on fixed-length LM training-token exposure:
        global_step * per_device_train_batch_size * gradient_accumulation_steps * max_seq_length

    Because this script chunks the corpus into 256-token sequences, this is the
    practical counter for "how many training tokens the model has consumed".
    """
    def __init__(
        self,
        max_seq_length=256,
        checkpoint_intervals=None,
        output_dir="./babylm_checkpoints",
        tokenizer_asset_dir=".",
    ):
        self.max_seq_length = max_seq_length
        self.checkpoint_intervals = checkpoint_intervals or []
        self.output_dir = output_dir
        self.tokenizer_asset_dir = tokenizer_asset_dir

        self.total_tokens_seen = 0
        self.checkpoints_saved = set()

        os.makedirs(self.output_dir, exist_ok=True)

    def _tokens_per_optimizer_step(self, args):
        return (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * self.max_seq_length
        )

    def _checkpoint_name(self, checkpoint_tokens: int) -> str:
        return f"chck_{checkpoint_tokens // 1_000_000}M"

    def _copy_tokenizer_assets(self, save_dir: str):
        """
        Copy tokenizer assets needed to reload/evaluate the NOOVERLAP model.

        model.save_pretrained() saves the model/config, but your tokenizer is a
        custom object built from tokenizer_eng/nld/zho.json plus tokenizer_utilities.py.
        """
        asset_names = [
            "tokenizer_eng.json",
            "tokenizer_nld.json",
            "tokenizer_zho.json",
            "tokenizer_utilities.py",
        ]

        copied = []
        for name in asset_names:
            src = os.path.join(self.tokenizer_asset_dir, name)
            dst = os.path.join(save_dir, name)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied.append(name)
            else:
                logger.warning(f"Tokenizer asset not found, not copied: {src}")

        return copied

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_tokens: int):
        checkpoint_name = self._checkpoint_name(checkpoint_tokens)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save model/config in Hugging Face format for the evaluation pipeline.
        model.save_pretrained(save_dir, safe_serialization=True)

        # Save tokenizer assets for your custom NOOVERLAP tokenizer.
        copied_assets = self._copy_tokenizer_assets(save_dir)

        # Save training arguments and metadata for reproducibility.
        with open(os.path.join(save_dir, "training_args.json"), "w") as f:
            json.dump(args.to_dict(), f, indent=2)

        latest_log = state.log_history[-1] if state.log_history else {}
        metadata = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_tokens": checkpoint_tokens,
            "checkpoint_millions": checkpoint_tokens / 1_000_000,
            "actual_tokens_seen_estimate": self.total_tokens_seen,
            "global_step": state.global_step,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "loss": latest_log.get("loss", None),
            "learning_rate": latest_log.get("learning_rate", None),
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "tokens_per_optimizer_step": self._tokens_per_optimizer_step(args),
            "timestamp": datetime.now().isoformat(),
            "tokenizer_assets_copied": copied_assets,
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "note": (
                "This checkpoint was saved at a BabyLM exposure threshold. "
                "It is separate from Hugging Face Trainer recovery checkpoints."
            ),
        }

        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"{'=' * 70}")
        logger.info(f"Saved BabyLM evaluation checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Requested exposure threshold: {checkpoint_tokens:,} tokens")
        logger.info(f"Actual estimated exposure: {self.total_tokens_seen:,} tokens")
        logger.info(f"Global step: {state.global_step}")
        logger.info(f"Tokenizer assets copied: {copied_assets}")
        logger.info(f"{'=' * 70}")

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Initialize from state.global_step. This keeps the counter sane even if
        you later resume from a Hugging Face Trainer checkpoint.
        """
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
        """
        Called after each optimizer step.

        This saves BabyLM evaluation checkpoints directly, while the normal
        Trainer checkpoint logic continues independently from save_steps=500.
        """
        model = kwargs.get("model", None)
        if model is None:
            logger.warning("No model found in callback kwargs; cannot save BabyLM checkpoint.")
            return control

        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step

        # Save every BabyLM threshold crossed by this optimizer step.
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
    Save detailed checkpoint metadata every N steps for training recovery.
    Allows resuming from any step if training is interrupted.
    Saved to: checkpoints_detailed/checkpoint_step_{step:06d}.json
    """
    def __init__(self, checkpoint_dir: str = "./checkpoints_detailed", save_every_n_steps: int = 1000):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info = {}
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"DetailedCheckpointCallback: Saving recovery checkpoints every {save_every_n_steps} steps")
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        current_step = state.global_step
        
        if current_step % self.save_every_n_steps == 0 and current_step > 0:
            self._save_checkpoint_info(current_step, state, args)
    
    def _save_checkpoint_info(self, step: int, state, args):
        """Save detailed checkpoint information"""
        latest_log = state.log_history[-1] if state.log_history else {}
        
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": float(latest_log.get("loss", None)),
            "learning_rate": float(latest_log.get("learning_rate", args.learning_rate)),
            "epoch": float(state.epoch),
            "total_steps": state.max_steps,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }
        
        # Calculate progress
        progress_pct = (step / state.max_steps * 100) if state.max_steps > 0 else 0
        checkpoint_data["progress_percent"] = progress_pct
        
        # Save individual checkpoint file
        checkpoint_file = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json")
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Update master log
        self.checkpoint_info[step] = checkpoint_data
        master_log_file = os.path.join(self.checkpoint_dir, "checkpoint_log.json")
        with open(master_log_file, "w") as f:
            json.dump(self.checkpoint_info, f, indent=2)
        
        logger.info(
            f"Recovery checkpoint {step}: Loss={checkpoint_data['loss']:.4f} | "
            f"Epoch={checkpoint_data['epoch']:.1f} | "
            f"Progress={progress_pct:.1f}%"
        )


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(vocab_size: int) -> GPT2LMHeadModel:
    """Create GPT-2 model with specified configuration"""
    logger.info("Creating GPT-2 model")

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=TRAINING_CONFIG["hidden_size"],
        n_layer=TRAINING_CONFIG["num_hidden_layers"],
        n_head=TRAINING_CONFIG["num_attention_heads"],
        n_positions=TRAINING_CONFIG["data"]["max_seq_length"],
        bos_token_id=None,
        eos_token_id=None,
    )

    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created model with {num_params:,} parameters")

    return model


# ============================================================================
# TRAINING SETUP
# ============================================================================

def setup_training(model: GPT2LMHeadModel, train_dataset: Dataset) -> tuple:
    """Setup training arguments and trainer"""
    logger.info("Setting up training")

    training_args = TrainingArguments(
        output_dir=TRAINING_CONFIG["output_dir"],
        num_train_epochs=TRAINING_CONFIG["training"]["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["training"]["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["training"]["gradient_accumulation_steps"],
        logging_steps=TRAINING_CONFIG["training"]["logging_steps"],
        save_strategy="steps",
        save_steps=500,  # Save model checkpoint every 500 steps (for recovery)
        learning_rate=TRAINING_CONFIG["training"]["learning_rate"],
        warmup_steps=TRAINING_CONFIG["training"]["warmup_steps"],
        weight_decay=TRAINING_CONFIG["training"]["weight_decay"],
        max_grad_norm=TRAINING_CONFIG["training"]["max_grad_norm"],
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        seed=TRAINING_CONFIG["training"]["seed"],
        save_total_limit=5,  # Keep last 5 model checkpoints
    )

    # Custom data collator for padding
    def custom_collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Pad sequences to max length in batch
        max_len = max(len(x) for x in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, label in zip(input_ids, attention_mask, labels):
            padding_len = max_len - len(ids)
            padded_input_ids.append(ids + [0] * padding_len)
            padded_attention_mask.append(mask + [0] * padding_len)
            padded_labels.append(label + [-100] * padding_len)  # -100 is ignored in loss

        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_mask),
            "labels": torch.tensor(padded_labels),
        }

    # Initialize callbacks
    token_callback = TokenCounterCallback(
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
        checkpoint_intervals=TRAINING_CONFIG["checkpoint_intervals"],
        output_dir=TRAINING_CONFIG["babylm_checkpoint_dir"],
        tokenizer_asset_dir=".",
    )
    
    detailed_callback = DetailedCheckpointCallback(save_every_n_steps=1000)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_collate_fn,
        callbacks=[token_callback, detailed_callback],
    )

    logger.info("="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"Total epochs: {TRAINING_CONFIG['training']['num_epochs']}")
    logger.info(f"Total examples: {len(train_dataset):,}")
    logger.info(f"Sequence length: {TRAINING_CONFIG['data']['max_seq_length']}")
    logger.info(f"BabyLM competition checkpoints: {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"Recovery checkpoints: Every 1,000 steps")
    logger.info(f"Normal Trainer checkpoint dir: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM evaluation checkpoint dir: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info("="*70)

    return trainer, token_callback


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    logger.info("="*70)
    logger.info("BabyLM 2026 Multilingual Track - NOOVERLAP Tokenizer Training")
    logger.info("="*70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Load tokenizers
    logger.info("\n[1/6] Loading tokenizers...")
    tokenizers = load_monolingual_tokenizers()
    nooverlap_tokenizer = create_nooverlap_tokenizer(tokenizers)
    
    # Load datasets
    logger.info("\n[2/6] Loading datasets with byte-premium adjustment...")
    datasets = load_training_datasets(adjusted_budget_per_lang=33_333_333)
    
    # Prepare tagged dataset
    logger.info("\n[3/6] Preparing tagged dataset...")
    language_tags = {"eng": "<EN>", "nld": "<NL>", "zho": "<ZH>"}
    train_dataset = prepare_tagged_dataset(datasets, language_tags)
    
    # Preprocess
    logger.info("\n[4/6] Preprocessing dataset...")
    train_dataset = preprocess_dataset(
        train_dataset,
        nooverlap_tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"]
    )
    
    # Create model
    logger.info("\n[5/6] Creating model...")
    vocab_size = nooverlap_tokenizer.get_vocab_size()
    model = create_model(vocab_size)
    
    # Setup training
    logger.info("\n[6/6] Setting up training...")
    trainer, token_callback = setup_training(model, train_dataset)
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    logger.info(f"Target: 100M tokens (BabyLM 2026 Strict Track)")
    logger.info(f"Competition checkpoints: {len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"Recovery checkpoints: Every 1,000 steps to ./checkpoints_detailed/")
    logger.info(f"Normal Trainer checkpoints: {TRAINING_CONFIG['output_dir']}/checkpoint-*")
    logger.info(f"BabyLM evaluation checkpoints: {TRAINING_CONFIG['babylm_checkpoint_dir']}/chck_*")
    logger.info("="*70 + "\n")
    
    trainer.train()
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"BabyLM evaluation checkpoints saved: {len(token_callback.checkpoints_saved)}/{len(TRAINING_CONFIG['checkpoint_intervals'])}")
    logger.info(f"Total tokens trained: {token_callback.total_tokens_seen:,}")
    logger.info(f"Final model: {TRAINING_CONFIG['output_dir']}/final")
    logger.info("="*70)

    # Save final model
    model.save_pretrained(f"{TRAINING_CONFIG['output_dir']}/final")
    logger.info(f"✓ Final model saved to {TRAINING_CONFIG['output_dir']}/final")

    # List all normal Hugging Face Trainer recovery checkpoints
    import os
    checkpoint_dir = TRAINING_CONFIG['output_dir']
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')])
        logger.info(f"\n✓ Saved {len(checkpoints)} model recovery checkpoints in {checkpoint_dir}")
        logger.info(f"Recovery checkpoints: {checkpoints}")

    # List all BabyLM evaluation checkpoints
    babylm_dir = TRAINING_CONFIG["babylm_checkpoint_dir"]
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted([d for d in os.listdir(babylm_dir) if d.startswith("chck_")])
        logger.info(f"✓ Saved {len(babylm_checkpoints)} BabyLM evaluation checkpoints in {babylm_dir}")
        logger.info(f"BabyLM evaluation checkpoints: {babylm_checkpoints}")


if __name__ == "__main__":
    main()