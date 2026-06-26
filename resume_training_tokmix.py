import os
import re
import logging
from typing import Optional

import torch
from transformers import set_seed

from train_tokmix_fixed import (
    TRAINING_CONFIG,
    load_tokmix_tokenizer,
    load_training_datasets,
    prepare_mixed_dataset,
    preprocess_dataset,
    create_model,
    setup_training,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_latest_trainer_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find latest Hugging Face Trainer checkpoint.

    Valid examples:
        ./model_tokmix/checkpoint-500
        ./model_tokmix/checkpoint-1000
    """
    if not os.path.exists(output_dir):
        return None

    checkpoints = []

    for name in os.listdir(output_dir):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if match is None:
            continue

        checkpoint_path = os.path.join(output_dir, name)
        if not os.path.isdir(checkpoint_path):
            continue

        step = int(match.group(1))
        checkpoints.append((step, checkpoint_path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def main() -> None:
    set_seed(TRAINING_CONFIG["training"]["seed"])

    logger.info("=" * 70)
    logger.info("Resuming BabyLM TOKMIX training")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Output dir: {TRAINING_CONFIG['output_dir']}")

    latest_checkpoint = find_latest_trainer_checkpoint(TRAINING_CONFIG["output_dir"])

    if latest_checkpoint is None:
        raise FileNotFoundError(
            f"No Hugging Face Trainer checkpoint found in {TRAINING_CONFIG['output_dir']}.\n"
            "Expected something like:\n"
            "  model_tokmix/checkpoint-500\n"
            "  model_tokmix/checkpoint-1000\n\n"
            "Do not use babylm_checkpoints_tokmix/chck_* for exact resume; those are evaluation checkpoints."
        )

    logger.info(f"Latest Trainer checkpoint: {latest_checkpoint}")

    logger.info("[1/6] Loading TOKMIX tokenizer")
    tokenizer = load_tokmix_tokenizer()

    logger.info("[2/6] Loading datasets")
    datasets = load_training_datasets(
        adjusted_budget_per_lang=TRAINING_CONFIG["data"]["adjusted_budget_per_lang"]
    )

    logger.info("[3/6] Preparing mixed dataset")
    train_dataset = prepare_mixed_dataset(datasets)

    logger.info("[4/6] Tokenizing/chunking dataset")
    train_dataset = preprocess_dataset(
        train_dataset,
        tokenizer,
        max_seq_length=TRAINING_CONFIG["data"]["max_seq_length"],
    )

    logger.info("[5/6] Creating model")
    model = create_model(len(tokenizer), tokenizer)

    logger.info("[6/6] Setting up Trainer")
    trainer, token_callback = setup_training(model, train_dataset, tokenizer)

    logger.info("=" * 70)
    logger.info(f"Calling trainer.train(resume_from_checkpoint={latest_checkpoint})")
    logger.info("=" * 70)

    trainer.train(resume_from_checkpoint=latest_checkpoint)

    logger.info("=" * 70)
    logger.info("TOKMIX resumed training complete")
    logger.info("=" * 70)
    logger.info(f"Total estimated token exposure: {token_callback.total_tokens_seen:,}")

    final_dir = os.path.join(TRAINING_CONFIG["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    logger.info(f"Saved final resumed model to: {final_dir}")


if __name__ == "__main__":
    main()