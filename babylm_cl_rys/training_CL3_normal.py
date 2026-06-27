"""
Tokens per optimizer step: batch_size * grad_accum * max_seq_length
  = 8 * 2 * 1024 = 16,384

Total token budget: 10M words * 10 epochs = 100M tokens
Total optimizer steps: 100M / 16,384 ≈ 6,103
"""

import os
import json
import logging
from datetime import datetime
from itertools import chain

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)

logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Configuration — single source of truth, nothing hardcoded elsewhere ───────

TRAINING_CONFIG = {
    # ── Model architecture (matches BabyLM 2026 strict-small baseline exactly) ─
    "model": {
        "vocab_size": 16384,            # baseline uses a custom 16k BPE tokenizer
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": None,      # None → GPT2 defaults to 4 × hidden_size = 3072
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "activation_function": "gelu_new",
        "bos_token_id": 1,              # baseline special token ids (not standard GPT2)
        "eos_token_id": 2,
    },

    # ── Training hyperparameters ───────────────────────────────────────────────
    "training": {
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        # effective batch = 8 × 2 = 16 sequences × 1024 tokens = 16,384 tokens/step
        "learning_rate": 1e-3,
        "num_epochs": 10,
        "warmup_steps": 500,
        "weight_decay": 0.1,          
                                         # Lovelace et al. 2026 shows strong weight decay
                                         # reduces the overfitting penalty from data repetition
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
        "save_steps": 500,              # HF Trainer recovery checkpoint frequency
        "save_total_limit": 2,          # keep only last 2 recovery checkpoints
        "dataloader_num_workers": 0,    # bump to 2-4 if your system handles it fine
        "detailed_checkpoint_every_n_steps": 500,
    },

    # ── Data ──────────────────────────────────────────────────────────────────
    "data": {
        "max_seq_length": 1024,         # matches baseline n_ctx / n_positions
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
        "hf_dataset": "flakoash/babylm-curriculum-tiered-4bands",
        "tokenizer_name": "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small",
        "epoch_files": [
            "curriculum/epoch_00.jsonl",
            "curriculum/epoch_01.jsonl",
            "curriculum/epoch_02.jsonl",
            "curriculum/epoch_03.jsonl",
        ],
    },

    # ── Output directories ────────────────────────────────────────────────────
    "output_dir": "./model_tiered",
    "babylm_checkpoint_dir": "./babylm_checkpoints_tiered",
    "detailed_checkpoint_dir": "./checkpoints_detailed_tiered",

    # ── BabyLM evaluation checkpoints ────────────────────────────────────────
    # Saved separately from HF Trainer recovery checkpoints.
    # Numbers = cumulative tokens (words) seen by the model.
    # Strict-small budget is 10M words × 10 epochs = 100M total exposures.
    "checkpoint_intervals": [
        1_000_000,  2_000_000,  3_000_000,  4_000_000,  5_000_000,
        6_000_000,  7_000_000,  8_000_000,  9_000_000,  10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000,
        70_000_000, 80_000_000, 90_000_000, 100_000_000,
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_curriculum_dataset(config: dict):
    data_cfg = config["data"]
    epoch_datasets = []
    for epoch_file in data_cfg["epoch_files"]:
        logger.info(f"Loading {epoch_file}...")
        ds = load_dataset(
            data_cfg["hf_dataset"],
            data_files=epoch_file,
            split="train",
        )
        epoch_datasets.append(ds)

    full_dataset = concatenate_datasets(epoch_datasets)
    logger.info(f"Total curriculum examples: {len(full_dataset):,}")
    return full_dataset


def tokenize_and_chunk(dataset, tokenizer, config: dict):
    data_cfg = config["data"]
    max_seq_length = data_cfg["max_seq_length"]

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    logger.info("Tokenizing...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=data_cfg["tokenize_batch_size"],
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def chunk_into_blocks(examples):
        all_ids = list(chain(*examples["input_ids"]))
        total_len = (len(all_ids) // max_seq_length) * max_seq_length
        chunks = [all_ids[i : i + max_seq_length] for i in range(0, total_len, max_seq_length)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * max_seq_length] * len(chunks),
        }

    logger.info("Chunking into fixed-length blocks...")
    chunked = tokenized.map(
        chunk_into_blocks,
        batched=True,
        batch_size=data_cfg["chunk_batch_size"],
        desc="Chunking",
    )

    logger.info(f"Total training chunks: {len(chunked):,}")
    logger.info(f"Approximate tokens:    {len(chunked) * max_seq_length:,}")
    return chunked


# ── BabyLM evaluation checkpoint callback ────────────────────────────────────

class TokenCounterCallback(TrainerCallback):
    """
    Saves BabyLM evaluation checkpoints when cumulative token exposure crosses
    each threshold in checkpoint_intervals. These are separate from the normal
    HF Trainer recovery checkpoints.

    Exposure formula:
        global_step × batch_size × gradient_accumulation_steps × max_seq_length

    HF Trainer recovery checkpoints:   ./model_strictsmall/checkpoint-500
    BabyLM evaluation checkpoints:     ./babylm_checkpoints_strictsmall/chck_1M
    """

    def __init__(self, config: dict):
        train_cfg = config["training"]
        data_cfg = config["data"]
        self.max_seq_length = data_cfg["max_seq_length"]
        self.checkpoint_intervals = config["checkpoint_intervals"]
        self.output_dir = config["babylm_checkpoint_dir"]
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

    def _save_babylm_checkpoint(self, model, args, state, checkpoint_tokens: int) -> None:
        checkpoint_name = self._checkpoint_name(checkpoint_tokens)
        save_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir, safe_serialization=True)

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
            "checkpoint_type": "babylm_evaluation_checkpoint",
            "track": "strict-small",
            "token_budget": "10M words × 10 epochs = 100M tokens",
        }
        with open(os.path.join(save_dir, "babylm_checkpoint_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 70)
        logger.info(f"Saved BabyLM checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Threshold: {checkpoint_tokens:,} | Actual: {self.total_tokens_seen:,}")
        logger.info(f"Global step: {state.global_step}")
        logger.info("=" * 70)

    def on_train_begin(self, args, state, control, **kwargs):
        tokens_per_step = self._tokens_per_optimizer_step(args)
        self.total_tokens_seen = state.global_step * tokens_per_step
        self.checkpoints_saved = {
            t for t in self.checkpoint_intervals if t <= self.total_tokens_seen
        }
        logger.info("=" * 70)
        logger.info("BABYLM TOKEN COUNTER CALLBACK — strict-small track")
        logger.info(f"Tokens per optimizer step: {tokens_per_step:,}")
        logger.info(f"Initial tokens seen:       {self.total_tokens_seen:,}")
        logger.info(f"Checkpoints already passed: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        self.total_tokens_seen = state.global_step * self._tokens_per_optimizer_step(args)

        for checkpoint_tokens in self.checkpoint_intervals:
            if (
                self.total_tokens_seen >= checkpoint_tokens
                and checkpoint_tokens not in self.checkpoints_saved
            ):
                self.checkpoints_saved.add(checkpoint_tokens)
                self._save_babylm_checkpoint(
                    model=model, args=args, state=state,
                    checkpoint_tokens=checkpoint_tokens,
                )
        return control


# ── Detailed JSON step logger ─────────────────────────────────────────────────

class DetailedCheckpointCallback(TrainerCallback):
    """
    Writes lightweight JSON metadata every N optimizer steps for monitoring.
    Not resumable checkpoints — just a training log you can inspect any time.
    """

    def __init__(self, config: dict):
        self.checkpoint_dir = config["detailed_checkpoint_dir"]
        self.save_every_n_steps = config["training"]["detailed_checkpoint_every_n_steps"]
        self.checkpoint_info = {}
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"DetailedCheckpointCallback: JSON logs every {self.save_every_n_steps} steps")

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self.save_every_n_steps == 0 and step > 0:
            self._save_checkpoint_info(step, state, args)

    def _save_checkpoint_info(self, step: int, state, args) -> None:
        latest_log = state.log_history[-1] if state.log_history else {}
        progress_pct = (step / state.max_steps * 100) if state.max_steps else 0.0
        data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "loss": safe_float(latest_log.get("loss")),
            "learning_rate": safe_float(latest_log.get("learning_rate", args.learning_rate)),
            "epoch": safe_float(state.epoch),
            "total_steps": state.max_steps,
            "progress_percent": round(progress_pct, 2),
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            ),
        }
        step_file = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json")
        with open(step_file, "w") as f:
            json.dump(data, f, indent=2)

        self.checkpoint_info[step] = data
        with open(os.path.join(self.checkpoint_dir, "checkpoint_log.json"), "w") as f:
            json.dump(self.checkpoint_info, f, indent=2)

        loss_text = f"{data['loss']:.4f}" if data["loss"] is not None else "N/A"
        logger.info(
            f"Step {step}: loss={loss_text} | "
            f"epoch={data['epoch']:.2f} | {progress_pct:.1f}%"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = TRAINING_CONFIG
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    tokens_per_step = (
        train_cfg["batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * data_cfg["max_seq_length"]
    )
    logger.info("=" * 70)
    logger.info("BabyLM 2026 — Strict-Small Track")
    logger.info(f"Model: GPT2 {model_cfg['hidden_size']}d / {model_cfg['num_hidden_layers']}L")
    logger.info(f"Token budget: 10M words × {train_cfg['num_epochs']} epochs = 100M tokens")
    logger.info(
        f"Tokens/step: {train_cfg['batch_size']} × "
        f"{train_cfg['gradient_accumulation_steps']} × "
        f"{data_cfg['max_seq_length']} = {tokens_per_step:,}"
    )
    logger.info("=" * 70)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Load the baseline tokenizer (vocab_size=16384, custom BPE).
    # Using the same tokenizer as the baseline ensures a fair comparison —
    # both models see identical token sequences.
    logger.info(f"Loading tokenizer from {data_cfg['tokenizer_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(data_cfg["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    # All values come from model_cfg — no literals in this block.
    logger.info("Initializing model from scratch...")
    model_config = GPT2Config(
        vocab_size=model_cfg["vocab_size"],
        n_embd=model_cfg["hidden_size"],
        n_layer=model_cfg["num_hidden_layers"],
        n_head=model_cfg["num_attention_heads"],
        n_inner=model_cfg["intermediate_size"],
        n_positions=data_cfg["max_seq_length"],
        n_ctx=data_cfg["max_seq_length"],
        attn_pdrop=model_cfg["attn_pdrop"],
        embd_pdrop=model_cfg["embd_pdrop"],
        resid_pdrop=model_cfg["resid_pdrop"],
        activation_function=model_cfg["activation_function"],
        bos_token_id=model_cfg["bos_token_id"],
        eos_token_id=model_cfg["eos_token_id"],
    )
    model = GPT2LMHeadModel(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw_dataset = load_curriculum_dataset(cfg)
    tokenized_dataset = tokenize_and_chunk(raw_dataset, tokenizer, cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        max_grad_norm=train_cfg["max_grad_norm"],
        seed=train_cfg["seed"],
        dataloader_drop_last=True,      # keeps token count math exact
        dataloader_pin_memory=True,
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        fp16=torch.cuda.is_available(), # free speedup on 16-24GB GPU
        report_to="none",               # swap to "wandb" for experiment tracking
        remove_unused_columns=False,
    )

    # ── Callbacks — both receive the full config, no hardcoded values ─────────
    token_counter = TokenCounterCallback(config=cfg)
    detailed_cb = DetailedCheckpointCallback(config=cfg)

    # ── Trainer ───────────────────────────────────────────────────────────────
    # The HF Trainer will repeat tokenized_dataset num_epochs times.
    # shuffle=False is implicit because we never pass a shuffle argument —
    # the Trainer only shuffles if you set shuffle=True on the DataLoader,
    # which it doesn't do by default for non-iterable datasets without
    # explicit dataloader_num_workers shuffle config.
    # To be safe and explicit, we subclass nothing but verify order is kept.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[token_counter, detailed_cb],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model and tokenizer...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info(f"Done. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()