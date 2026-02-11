import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, get_linear_schedule_with_warmup

from src.data_utils import AnswersDataset
from src.model_utils import build_model

import wandb


def main():
    train_path = "data/new_training.csv"
    val_path = "data/new_validation (1).csv"
    test_path = "data/new_test (2).csv"
    model_name = "allenai/longformer-base-4096"
    max_len = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_defaults = {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "model_name": model_name,
        "max_len": max_len,
        "train_batch_size": 2,
        "val_batch_size": 2,
        "epochs": 5,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "fusion_mode": "concat",
        "device": str(device),
    }

    wandb.init(project="research_longformer", config=config_defaults)
    config = wandb.config

    train_bs = config.train_batch_size
    val_bs = config.val_batch_size
    epochs = config.epochs
    lr = config.learning_rate
    warmup_ratio = config.warmup_ratio
    weight_decay = config.weight_decay

    best_val_acc = -1.0
    best_ckpt_path = "best_model.pt"

    train_df = pd.read_csv(train_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()
    val_df = pd.read_csv(val_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()
    test_df = pd.read_csv(test_path).dropna(subset=["ResponseText.x", "ground_truth"]).copy()

    train_df["ground_truth"] = train_df["ground_truth"].astype(int)
    val_df["ground_truth"] = val_df["ground_truth"].astype(int)
    test_df["ground_truth"] = test_df["ground_truth"].astype(int)

    min_label = int(train_df["ground_truth"].min())
    if min_label != 0:
        train_df["ground_truth"] -= min_label
        val_df["ground_truth"] -= min_label
        test_df["ground_truth"] -= min_label

    num_labels = int(train_df["ground_truth"].nunique())

    train_df["question"] = ""
    train_df["model_answer"] = ""
    val_df["question"] = ""
    val_df["model_answer"] = ""
    test_df["question"] = ""
    test_df["model_answer"] = ""

    tok = LongformerTokenizerFast.from_pretrained(model_name)

    #train_ds = AnswersDataset(train_df, tok, max_len=max_len)
    #val_ds = AnswersDataset(val_df, tok, max_len=max_len)
    #test_ds = AnswersDataset(test_df, tok, max_len=max_len)

    train_ds = AnswersDataset(
        train_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="model_answer",
        question_col="question",
        label_col="ground_truth",
        max_len=max_len,
        fusion_mode="concat"
    )

    val_ds = AnswersDataset(
        val_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="model_answer",
        question_col="question",
        label_col="ground_truth",
        max_len=max_len,
        fusion_mode="concat"
    )

    test_ds = AnswersDataset(
        test_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="model_answer",
        question_col="question",
        label_col="ground_truth",
        max_len=max_len,
        fusion_mode="concat"
    )

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=val_bs, shuffle=False)

    model = build_model(model_name, num_labels=num_labels, fusion_mode="concat").to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_attention_mask=batch.get("global_attention_mask", None),
            )
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()

            wandb.log({"train_batch_loss": loss.item(), "learning_rate": optim.param_groups[0]["lr"]})

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / max(num_train_batches, 1)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

        model.eval()
        total = 0
        correct = 0
        val_loss_total = 0.0
        num_val_batches = 0

        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    global_attention_mask=batch.get("global_attention_mask", None),
                )

                val_loss_total += out["loss"].item()
                num_val_batches += 1

                preds = out["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        val_loss = val_loss_total / max(num_val_batches, 1)
        val_acc = correct / total if total else 0.0

        wandb.log({"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc})

        print(f"epoch {epoch}  train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    total = 0
    correct = 0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_attention_mask=batch.get("global_attention_mask", None),
            )

            preds = out["logits"].argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"].cpu())

    test_acc = correct / total if total else 0.0
    print(f"\nFinal TEST accuracy: {test_acc:.4f}")

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    corr = None
    if all_preds.std() != 0 and all_labels.std() != 0:
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
        print(f"Correlation between predictions and labels: {corr:.4f}")
    else:
        print("Correlation between predictions and labels: undefined")

    wandb.log({"test_acc": test_acc, "test_corr": float(corr) if corr is not None else None})


if __name__ == "__main__":
    main()

