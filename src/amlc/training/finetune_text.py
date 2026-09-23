"""LoRA fine-tune of an HF encoder/decoder for regression or classification on text.

Restartable: checkpoints every --save-steps to --output-dir and auto-resumes from the
latest one. On SageMaker managed spot, point --output-dir at /opt/ml/checkpoints (synced
to S3 by SageMaker); on Kaggle, at /kaggle/working/ckpt and re-attach the output next session.

    python -m amlc.training.finetune_text --train data/train.parquet --text-col catalog_content \
        --label-col price --task reg --target log1p --model microsoft/deberta-v3-base \
        --output-dir checkpoints/deberta-lora --epochs 2 --bf16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def latest_checkpoint(d: str | Path) -> str | None:
    cks = sorted(Path(d).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--text-col", required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--task", choices=["reg", "clf"], default="reg")
    p.add_argument("--target", choices=["none", "log1p"], default="none")
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA for decoder LLMs")
    a = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    from amlc.data.io import read_table

    df = read_table(a.train).select([a.text_col, a.label_col]).drop_nulls(a.label_col)
    texts = df[a.text_col].fill_null("").to_list()
    labels = df[a.label_col].to_numpy()
    if a.task == "reg":
        labels = labels.astype(np.float32)
        if a.target == "log1p":
            labels = np.log1p(labels)
        num_labels, label_map = 1, None
    else:
        classes = sorted(set(labels.tolist()))
        label_map = {c: i for i, c in enumerate(classes)}
        labels = np.array([label_map[x] for x in labels])
        num_labels = len(classes)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"num_labels": num_labels}
    if a.task == "reg":
        kw["problem_type"] = "regression"
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, **kw)
    model.config.pad_token_id = tok.pad_token_id

    if not a.full:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if a.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=a.lora_r, lora_alpha=2 * a.lora_r,
                                                 lora_dropout=0.05, target_modules="all-linear"))
        model.print_trainable_parameters()

    ds = Dataset.from_dict({"text": texts, "labels": labels.tolist()})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_length), batched=True,
                remove_columns=["text"])
    split = ds.train_test_split(test_size=a.val_frac, seed=42)

    args = TrainingArguments(
        output_dir=a.output_dir, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        gradient_accumulation_steps=a.grad_accum, warmup_ratio=0.05, lr_scheduler_type="cosine",
        eval_strategy="steps", eval_steps=a.save_steps, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=2, logging_steps=50, bf16=a.bf16, fp16=a.fp16, report_to=[],
        dataloader_num_workers=4, load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"],
                      data_collator=DataCollatorWithPadding(tok))
    trainer.train(resume_from_checkpoint=latest_checkpoint(a.output_dir))
    trainer.save_model(str(Path(a.output_dir) / "final"))  # LoRA: saves only the small adapter
    tok.save_pretrained(str(Path(a.output_dir) / "final"))
    import json

    (Path(a.output_dir) / "final" / "amlc_labels.json").write_text(json.dumps(
        {"task": a.task, "target": a.target, "classes": list(label_map) if label_map else None}, default=str))
    print("saved", Path(a.output_dir) / "final")


if __name__ == "__main__":
    main()
