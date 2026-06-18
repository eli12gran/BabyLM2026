#!/usr/bin/env python3
"""
Resume BabyLM NOOVERLAP training from the latest Hugging Face Trainer checkpoint.

Place this file in the same directory as train.py, then run:

    python resume_training.py

It will:
1. Find the latest checkpoint in ./model_nooverlap, e.g. checkpoint-16000.
2. Rebuild the tokenizer, dataset, model, Trainer, and callbacks using train.py.
3. Resume training with trainer.train(resume_from_checkpoint=...).

This uses the real Trainer checkpoint, not checkpoints_detailed/*.json.
"""

import os
import re
import json
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

from train import (
    TRAINING_CONFIG,
    logger,
    load_monolingual_tokenizers,
    create_nooverlap_tokenizer,
    load_training_datasets,
    prepare_tagged_dataset,
    preprocess_dataset,
    create_model,
    setup_training,
)


def checkpoint_step(path: str) -> int:
    """
    Extract the numeric step from a checkpoint directory name.
    Example: ./model_nooverlap/checkpoint-16000 -> 16000
    """
    name = os.path.basename(os.path.normpath(path))
    match = re.match(r"checkpoint-(\d+)$", name)
    if not match:
        return -1
    return int(match.group(1))


def find_latest_checkpoint(output_dir: str) -> str:
    """
    Find the latest valid Hugging Face Trainer checkpoint in output_dir.
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    last_checkpoint = get_last_checkpoint(str(output_path))

    if last_checkpoint is None:
        candidates = [
            str(p)
            for p in output_path.iterdir()
            if p.is_dir() and re.match(r"checkpoint-\d+$", p.name)
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint-* directories found in {output_dir}"
            )

        last_checkpoint = max(candidates, key=checkpoint_step)

    required_files = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    ]

    missing = [
        f for f in required_files
        if not os.path.exists(os.path.join(last_checkpoint, f))
    ]

    if missing:
        raise FileNotFoundError(
            f"Latest checkpoint is missing required Trainer files: {missing}\n"
            f"Checkpoint path: {last_checkpoint}"
        )

    return last_checkpoint


def print_checkpoint_summary(checkpoint_path: str):
    """
    Print useful checkpoint state before resuming.
    """
    state_path = os.path.join(checkpoint_path, "trainer_state.json")

    logger.info("=" * 70)
    logger.info("RESUME CHECKPOINT FOUND")
    logger.info(f"Path: {checkpoint_path}")

    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)

        logger.info(f"global_step: {state.get('global_step')}")
        logger.info(f"epoch: {state.get('epoch')}")
        logger.info(f"max_steps: {state.get('max_steps')}")
        logger.info(f"best_metric: {state.get('best_metric')}")
        logger.info(f"best_model_checkpoint: {state.get('best_model_checkpoint')}")

    logger.info("=" * 70)


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    output_dir = TRAINING_CONFIG["output_dir"]
    latest_checkpoint = find_latest_checkpoint(output_dir)
    print_checkpoint_summary(latest_checkpoint)

    logger.info("=" * 70)
    logger.info("REBUILDING TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    logger.info("\n[1/6] Loading tokenizers...")
    tokenizers = load_monolingual_tokenizers()
    nooverlap_tokenizer = create_nooverlap_tokenizer(tokenizers)

    logger.info("\n[2/6] Loading datasets with byte-premium adjustment...")
    datasets = load_training_datasets(adjusted_budget_per_lang=33_333_333)

    logger.info("\n[3/6] Preparing tagged dataset...")
    language_tags = {"eng": "<EN>", "nld": "<NL>", "zho": "<ZH>"}
    train_dataset = prepare_tagged_dataset(datasets, language_tags)

    logger.info("\n[4/6] Preprocessing dataset...")
    train_dataset = preprocess_dataset(
        train_dataset,
        nooverlap_tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
    )

    logger.info("\n[5/6] Creating model...")
    vocab_size = nooverlap_tokenizer.get_vocab_size()
    model = create_model(vocab_size)

    logger.info("\n[6/6] Setting up Trainer...")
    trainer, token_callback = setup_training(model, train_dataset)

    logger.info("=" * 70)
    logger.info("RESUMING TRAINING")
    logger.info(f"Resume checkpoint: {latest_checkpoint}")
    logger.info("=" * 70)

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    logger.info("\n" + "=" * 70)
    logger.info("RESUMED TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total tokens seen estimate: {token_callback.total_tokens_seen:,}")
    logger.info(f"Final model path: {TRAINING_CONFIG['output_dir']}/final")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    model.save_pretrained(final_dir)
    logger.info(f"Saved final model to: {final_dir}")

    recovery_dir = TRAINING_CONFIG["output_dir"]
    if os.path.exists(recovery_dir):
        recovery_checkpoints = sorted(
            [d for d in os.listdir(recovery_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        logger.info(f"Recovery checkpoints in {recovery_dir}: {recovery_checkpoints}")

    babylm_dir = TRAINING_CONFIG.get("babylm_checkpoint_dir", "./babylm_checkpoints")
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted(
            [d for d in os.listdir(babylm_dir) if d.startswith("chck_")]
        )
        logger.info(f"BabyLM evaluation checkpoints in {babylm_dir}: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
