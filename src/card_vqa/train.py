import argparse
import os

import matplotlib.pyplot as plt
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from card_vqa.config import load_config
from card_vqa.data import VQACollator, VQADataset
from card_vqa.model import build_lora_model, clear_memory, load_base_model, load_processor


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with LoRA for VQA.")
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.get("patch_peft_torchao_compat", False):
        # Workaround used in the original notebook; only enable if your runtime needs it.
        import peft.import_utils
        peft.import_utils.is_torchao_available = lambda: False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output_dir"], exist_ok=True)
    best_path = os.path.join(cfg["output_dir"], "best_lora_adapter")

    processor = load_processor(
        cfg["model_id"], cfg["min_pixels"], cfg["max_pixels"]
    )
    model = load_base_model(cfg["model_id"], device)
    model = build_lora_model(model, cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type="cosine",
        num_train_epochs=cfg["num_train_epochs"],
        fp16=(device == "cuda"),
        gradient_checkpointing=True,
        logging_steps=cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=VQADataset(cfg["train_json"], cfg["image_dir"], processor),
        eval_dataset=VQADataset(cfg["val_json"], cfg["image_dir"], processor),
        data_collator=VQACollator(processor),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["early_stopping_patience"]
            )
        ],
    )
    trainer.train()
    trainer.model.save_pretrained(best_path)
    processor.save_pretrained(best_path)

    val_losses = [
        entry["eval_loss"]
        for entry in trainer.state.log_history
        if "eval_loss" in entry
    ]
    if val_losses:
        plt.figure(figsize=(6, 3.5), dpi=120)
        plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s",
                 label="Validation loss")
        plt.scatter([val_losses.index(min(val_losses)) + 1], [min(val_losses)])
        plt.title(f"Validation Loss (Best: {min(val_losses):.4f})")
        plt.xlabel("Evaluation step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(cfg["output_dir"], "loss_curve.png")
        plt.savefig(plot_path)
        print(f"Saved loss curve: {plot_path}")

    del model, trainer
    clear_memory()
    print(f"Saved best adapter and processor to: {best_path}")


if __name__ == "__main__":
    main()
