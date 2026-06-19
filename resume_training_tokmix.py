#!/usr/bin/env python3
"""
Resume BabyLM TOKMIX training from the latest Hugging Face Trainer checkpoint.

Place this file in the same directory as your TOKMIX training script, then run:

    python resume_training_tokmix.py

Assumption:
    Your TOKMIX training script is named train_tokmix.py.

If your TOKMIX training script is named train.py instead, change:

    from train_tokmix import ...

to:

    from train import ...

What this script does:
1. Finds the latest Hugging Face Trainer checkpoint in TRAINING_CONFIG["output_dir"],
   for example:
       ./checkpoints_tokmix/checkpoint-16000
2. Rebuilds the tokenizer, dataset, model, Trainer, and callbacks using the same
   TOKMIX training functions.
3. Resumes with:
       trainer.train(resume_from_checkpoint=latest_checkpoint)

Important:
    This resumes from real Trainer checkpoints, not from checkpoints_detailed_tokmix/*.json.
"""

import os
import re
import json
from pathlib import Path

import torch
from transformers.trainer_utils import get_last_checkpoint

# If your training file is named train.py, replace train_tokmix with train.
from train_tokmix import (
    TRAINING_CONFIG,
    logger,
    load_tokmix_tokenizer,
    load_training_datasets,
    prepare_mixed_dataset,
    preprocess_dataset,
    create_model,
    setup_training,
)


# ---------------------------------------------------------------------------
# Directory alignment
# ---------------------------------------------------------------------------
# These should match the TOKMIX directories you want.
#
# If the interrupted run already saved checkpoints in ./model_tokmix,
# do NOT change output_dir here unless you also move/rename that directory.
#
# Recommended final TOKMIX layout:
#   ./checkpoints_tokmix/              Hugging Face Trainer checkpoints every 500 steps
#   ./babylm_checkpoints_tokmix/       BabyLM token-exposure checkpoints
#   ./checkpoints_detailed_tokmix/     JSON metadata checkpoints
#
TRAINING_CONFIG["output_dir"] = "./checkpoints_tokmix"
TRAINING_CONFIG["babylm_checkpoint_dir"] = "./babylm_checkpoints_tokmix"
TRAINING_CONFIG["detailed_checkpoint_dir"] = "./checkpoints_detailed_tokmix"


def checkpoint_step(path: str) -> int:
    """
    Extract the numeric step from a checkpoint directory name.

    Example:
        ./checkpoints_tokmix/checkpoint-16000 -> 16000
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
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}\n\n"
            "This means either training has not created checkpoints yet, or "
            "TRAINING_CONFIG['output_dir'] does not match the directory used "
            "during the interrupted run."
        )

    last_checkpoint = get_last_checkpoint(str(output_path))

    if last_checkpoint is None:
        candidates = [
            str(p)
            for p in output_path.iterdir()
            if p.is_dir() and re.match(r"checkpoint-\d+$", p.name)
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint-* directories found in {output_dir}.\n\n"
                "Remember: the every-500-step Trainer checkpoints are saved "
                "inside TRAINING_CONFIG['output_dir']."
            )

        last_checkpoint = max(candidates, key=checkpoint_step)

    required_files = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
    ]

    missing = [
        f
        for f in required_files
        if not os.path.exists(os.path.join(last_checkpoint, f))
    ]

    if missing:
        raise FileNotFoundError(
            f"Latest checkpoint is missing required Trainer files: {missing}\n"
            f"Checkpoint path: {last_checkpoint}\n\n"
            "This directory is probably not a full Hugging Face Trainer "
            "recovery checkpoint."
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
    logger.info("REBUILDING TOKMIX TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Trainer checkpoint directory: {TRAINING_CONFIG['output_dir']}")
    logger.info(f"BabyLM checkpoint directory: {TRAINING_CONFIG['babylm_checkpoint_dir']}")
    logger.info(f"Detailed metadata directory: {TRAINING_CONFIG['detailed_checkpoint_dir']}")

    # 1. Load TOKMIX tokenizer.
    logger.info("\n[1/6] Loading TOKMIX tokenizer...")
    tokenizer = load_tokmix_tokenizer()

    # 2. Reload the exact training data selection.
    logger.info("\n[2/6] Loading datasets with byte-premium adjustment...")
    datasets = load_training_datasets(
        adjusted_budget_per_lang=TRAINING_CONFIG["data"]["adjusted_budget_per_lang"]
    )

    # 3. Recreate mixed dataset.
    logger.info("\n[3/6] Preparing mixed dataset...")
    train_dataset = prepare_mixed_dataset(datasets)

    # 4. Re-tokenize and chunk.
    logger.info("\n[4/6] Preprocessing dataset...")
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
    )

    # 5. Recreate model architecture.
    logger.info("\n[5/6] Creating model...")
    vocab_size = len(tokenizer)
    model = create_model(vocab_size, tokenizer)

    # 6. Recreate Trainer and callbacks.
    logger.info("\n[6/6] Setting up Trainer...")
    trainer, token_callback = setup_training(model, train_dataset, tokenizer)

    logger.info("=" * 70)
    logger.info("RESUMING TOKMIX TRAINING")
    logger.info(f"Resume checkpoint: {latest_checkpoint}")
    logger.info("=" * 70)

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    logger.info("\n" + "=" * 70)
    logger.info("RESUMED TOKMIX TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total token exposures estimate: {token_callback.total_tokens_seen:,}")
    logger.info(f"Final model path: {TRAINING_CONFIG['output_dir']}/final")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    logger.info(f"Saved final model and TOKMIX tokenizer to: {final_dir}")

    recovery_dir = TRAINING_CONFIG["output_dir"]
    if os.path.exists(recovery_dir):
        recovery_checkpoints = sorted(
            [d for d in os.listdir(recovery_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        logger.info(f"Trainer recovery checkpoints in {recovery_dir}: {recovery_checkpoints}")

    babylm_dir = TRAINING_CONFIG["babylm_checkpoint_dir"]
    if os.path.exists(babylm_dir):
        babylm_checkpoints = sorted(
            [d for d in os.listdir(babylm_dir) if d.startswith("chck_")]
        )
        logger.info(f"BabyLM evaluation checkpoints in {babylm_dir}: {babylm_checkpoints}")


if __name__ == "__main__":
    main()
