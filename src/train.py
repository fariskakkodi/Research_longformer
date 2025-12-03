import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizerFast, get_linear_schedule_with_warmup
from src.data_utils import AnswersDataset
from src.model_utils import build_model

import wandb

def main():
    csv_path = "classifies_edited.csv"
    model_name = "allenai/longformer-base-4096"
    max_len = 512
    train_bs = 2
    val_bs = 2
    epochs = 3
    lr = 2e-5
    warmup_ratio = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="research_longformer",  # you can rename this project
        config={
            "csv_path": csv_path,
            "model_name": model_name,
            "max_len": max_len,
            "train_batch_size": train_bs,
            "val_batch_size": val_bs,
            "epochs": epochs,
            "learning_rate": lr,
            "warmup_ratio": warmup_ratio,
            "device": str(device),
        },
    )

    best_val_acc = -1.0
    best_ckpt_path = "best_model.pt"

    df = pd.read_csv(csv_path).dropna(subset=["student_answer", "label"]).copy()
    df["label"] = df["label"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    tok = LongformerTokenizerFast.from_pretrained(model_name)
    train_ds = AnswersDataset(train_df, tok, max_len=max_len)
    val_ds = AnswersDataset(val_df, tok, max_len=max_len)
    test_ds = AnswersDataset(test_df, tok, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=val_bs, shuffle=False)

    model = build_model(model_name, num_labels=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
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
            wandb.log(
                {
                    "train_batch_loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                }
            )

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = running_train_loss / max(num_train_batches, 1)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})

        # ===== VALIDATION LOOP (FIXED) =====
        model.eval()
        total = correct = 0
        val_loss_total = 0.0      # accumulate validation loss
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

                # NEW: accumulate loss and batch count
                val_loss_total += out["loss"].item()
                num_val_batches += 1

                preds = out["logits"].argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        # NEW: compute average val_loss
        val_loss = val_loss_total / max(num_val_batches, 1)
        val_acc = correct / total if total else 0.0  # keep only one line

        wandb.log(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"epoch {epoch}  train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    total = correct = 0
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
    if all_preds.std() == 0 or all_labels.std() == 0:
        print("Correlation between predictions and labels: undefined")
        corr = None
    else:
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
        print(f"Correlation between predictions and labels: {corr:.4f}")

    wandb.log(
        {
            "test_acc": test_acc,
            "test_corr": float(corr) if corr is not None else None,
        }
    )

if __name__ == "__main__":
    main()
