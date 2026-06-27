import os
import json
import shutil
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

# ── Configuration ─────────────────────────────────────────────────────────────

TRAINING_CONFIG = {
    # Model: scaled-down GPT2 (~7M parameters)
    # Rule of thumb: params ≈ tokens/10 → 100M token budget → ~7-10M params is right
    "model_name": "gpt2",
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 1024,  # 4× hidden_size, standard FFN ratio

    "training": {
        # batch 16 × grad_accum 4 × seq 128 = 8,192 tokens/step
        # on 16-24GB GPU this is very comfortable for a 7M-param model
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-3,       # slightly higher than 100M-track; small model
                                      # benefits from more aggressive LR
        "num_epochs": 10,           
        "warmup_steps": 500,         # ~4% of total steps; proportionally same as 100M-track
        "weight_decay": 0.01,
        "logging_steps": 100,
        "seed": 42,
        "max_grad_norm": 1.0,
        "save_steps": 500,          
        "save_total_limit": 2,       # keep only last 2 recovery checkpoints to save disk
    },

    "data": {
        "max_seq_length": 128,     
                                      # has short sentences, less padding waste
        "tokenize_batch_size": 1000,
        "chunk_batch_size": 1000,
        "hf_dataset": "flakoash/babylm-curriculum-sliding-window-4bands",
        "epoch_files": [
            "curriculum/epoch_00.jsonl",
            "curriculum/epoch_01.jsonl",
            "curriculum/epoch_02.jsonl",
            "curriculum/epoch_03.jsonl",
        ],
    },

    "output_dir": "./model_strictsmall",
    "babylm_checkpoint_dir": "./babylm_checkpoints_strictsmall",
    "detailed_checkpoint_dir": "./checkpoints_detailed_strictsmall",

    # BabyLM 2026 exposure checkpoints (words/tokens seen).
    # Same structure as 100M-track but budget stops at 10M.
    # 1M–10M in 1M steps, then nothing beyond (unlike 100M-track which continued to 100M)
    "checkpoint_intervals": [
        1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000,
        6_000_000, 7_000_000, 8_000_000, 9_000_000, 10_000_000,
        20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000,
        70_000_000, 80_000_000, 90_000_000, 100_000_000
    ],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Dataset loading (curriculum-ordered, no shuffle) ─────────────────────────

def load_curriculum_dataset(config: dict):
    """
    Load the 4 epoch files in order and concatenate them.
    This single ordered pass = 1 curriculum cycle.
    The Trainer will repeat it num_epochs times.

    shuffle=False is critical — the curriculum ordering IS the training signal.
    """
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

    # Concatenate in curriculum order (epoch_00 first, epoch_03 last)
    full_dataset = concatenate_datasets(epoch_datasets)
    logger.info(f"Total curriculum examples: {len(full_dataset):,}")
    return full_dataset


def tokenize_and_chunk(dataset, tokenizer, config: dict):
    """
    Tokenize all texts, then chunk into fixed-length blocks of max_seq_length.
    This is the standard causal LM preparation — no padding, no truncation,
    just clean fixed-size chunks with the remainder dropped.
    """
    max_seq_length = config["data"]["max_seq_length"]
    tokenize_batch = config["data"]["tokenize_batch_size"]
    chunk_batch = config["data"]["chunk_batch_size"]

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    logger.info("Tokenizing...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=tokenize_batch,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    def chunk_into_blocks(examples):
        # Flatten all token ids into one long sequence, then slice into blocks
        all_ids = list(chain(*examples["input_ids"]))
        total_len = (len(all_ids) // max_seq_length) * max_seq_length
        chunks = [all_ids[i : i + max_seq_length] for i in range(0, total_len, max_seq_length)]
        return {"input_ids": chunks, "attention_mask": [[1] * max_seq_length] * len(chunks)}

    logger.info("Chunking into fixed-length blocks...")
    chunked = tokenized.map(
        chunk_into_blocks,
        batched=True,
        batch_size=chunk_batch,
        desc="Chunking",
    )

    logger.info(f"Total training chunks: {len(chunked):,}")
    logger.info(f"Approximate tokens: {len(chunked) * max_seq_length:,}")
    return chunked


# ── Token counter callback (BabyLM evaluation checkpoints) ───────────────────

class TokenCounterCallback(TrainerCallback):
    """
    Saves BabyLM evaluation checkpoints when the model reaches exposure
    thresholds (measured in tokens/words seen), separate from the normal
    HF Trainer recovery checkpoints.

    Exposure = global_step × batch_size × grad_accum_steps × max_seq_length

    Recovery checkpoints (HF Trainer):     ./model_strictsmall/checkpoint-500
    BabyLM evaluation checkpoints:         ./babylm_checkpoints_strictsmall/chck_1M
    """

    def __init__(
        self,
        max_seq_length: int = 128,
        checkpoint_intervals=None,
        output_dir: str = "./babylm_checkpoints_strictsmall",
    ):
        self.max_seq_length = max_seq_length
        self.checkpoint_intervals = checkpoint_intervals or []
        self.output_dir = output_dir
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
        logger.info(f"Saved BabyLM evaluation checkpoint: {checkpoint_name}")
        logger.info(f"Path: {save_dir}")
        logger.info(f"Requested exposure threshold: {checkpoint_tokens:,} tokens")
        logger.info(f"Actual estimated exposure:    {self.total_tokens_seen:,} tokens")
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
        logger.info(f"Initial tokens seen: {self.total_tokens_seen:,}")
        logger.info(f"Checkpoints already passed: {len(self.checkpoints_saved)}")
        logger.info("=" * 70)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
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


# ── Detailed JSON step callback ───────────────────────────────────────────────

class DetailedCheckpointCallback(TrainerCallback):
    """
    Saves lightweight JSON metadata every N optimizer steps for monitoring.
    Not resumable checkpoints — just a training log you can inspect.
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints_detailed_strictsmall",
        save_every_n_steps: int = 500,  # more frequent than 100M-track since total steps are fewer
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.checkpoint_info = {}
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"DetailedCheckpointCallback: JSON logs every {save_every_n_steps} steps")

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
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        }
        step_file = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step:06d}.json")
        with open(step_file, "w") as f:
            json.dump(data, f, indent=2)

        self.checkpoint_info[step] = data
        with open(os.path.join(self.checkpoint_dir, "checkpoint_log.json"), "w") as f:
            json.dump(self.checkpoint_info, f, indent=2)

        loss_text = f"{data['loss']:.4f}" if data["loss"] is not None else "N/A"
        logger.info(f"Step {step}: loss={loss_text} | epoch={data['epoch']:.2f} | {progress_pct:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = TRAINING_CONFIG
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    logger.info("=" * 70)
    logger.info("BabyLM 2026 — Strict-Small Track")
    logger.info(f"Model: GPT2 custom ({cfg['hidden_size']}d, {cfg['num_hidden_layers']}L)")
    logger.info(f"Token budget: 10M words × {train_cfg['num_epochs']} epochs = 100M tokens")
    logger.info(f"Tokens/step: {train_cfg['batch_size']} × {train_cfg['gradient_accumulation_steps']} × {data_cfg['max_seq_length']} = "
                f"{train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps'] * data_cfg['max_seq_length']:,}")
    logger.info("=" * 70)

    # Tokenizer — standard GPT2 BPE, add pad token
    logger.info("Loading GPT2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

    # Model — custom small config
    logger.info("Initializing model from scratch...")
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=cfg["hidden_size"],
        n_layer=cfg["num_hidden_layers"],
        n_head=cfg["num_attention_heads"],
        n_inner=cfg["intermediate_size"],
        n_positions=data_cfg["max_seq_length"],  # match seq length exactly
        n_ctx=data_cfg["max_seq_length"],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Dataset — load in curriculum order, no shuffle
    raw_dataset = load_curriculum_dataset(cfg)
    tokenized_dataset = tokenize_and_chunk(raw_dataset, tokenizer, cfg)

    # Data collator for causal LM (labels = input_ids, shifted internally by model)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked
    )

    # Training arguments
    # dataloader_num_workers=0 avoids multiprocessing issues on some systems;
    # bump to 2-4 if your machine handles it fine
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
        # Never shuffle — curriculum order must be preserved
        dataloader_drop_last=True,
        # Use fp16 on 16-24GB GPU for speed and memory savings
        fp16=torch.cuda.is_available(),
        report_to="none",           # swap to "wandb" if you want experiment tracking
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # Callbacks
    token_counter = TokenCounterCallback(
        max_seq_length=data_cfg["max_seq_length"],
        checkpoint_intervals=cfg["checkpoint_intervals"],
        output_dir=cfg["babylm_checkpoint_dir"],
    )
    detailed_cb = DetailedCheckpointCallback(
        checkpoint_dir=cfg["detailed_checkpoint_dir"],
        save_every_n_steps=500,
    )

    # Trainer
    # shuffle_train_dataset=False is the key setting — without this the Trainer
    # would randomize the order between epochs, destroying the curriculum
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[token_counter, detailed_cb],
    )

    # Disable internal dataset shuffling — curriculum order is sacred
    trainer.train_dataset = tokenized_dataset
    training_args.dataloader_pin_memory = True

    logger.info("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    logger.info("Saving final model...")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    logger.info(f"Done. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()