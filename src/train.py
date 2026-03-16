import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import LongformerTokenizerFast, get_linear_schedule_with_warmup

from src.data_utils import AnswersDataset
from src.model_utils import build_model

import wandb

import torch.nn.functional as F
import os
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    train_path = "data/train_new_data.csv"
    val_path = "data/validation_new_data.csv"
    test_path = "data/test_new_data.csv"
    model_name = "allenai/longformer-base-4096"
    max_len = 4096

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_defaults = {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "model_name": model_name,
        "max_len": max_len,
        "train_batch_size": 2,
        "val_batch_size": 2,
        "epochs": 3,
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


    tok = LongformerTokenizerFast.from_pretrained(model_name)

    #train_ds = AnswersDataset(train_df, tok, max_len=max_len)
    #val_ds = AnswersDataset(val_df, tok, max_len=max_len)
    #test_ds = AnswersDataset(test_df, tok, max_len=max_len)

    train_ds = AnswersDataset(
        train_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="Model_Answer",
        question_col="Question",
        #rubric_col="Rubric",
        label_col="ground_truth",
        max_len=max_len,
        fusion_mode="concat"
    )

    val_ds = AnswersDataset(
        val_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="Model_Answer",
        question_col="Question",
        #rubric_col="Rubric",
        label_col="ground_truth",
        max_len=max_len,
        fusion_mode="concat"
    )

    test_ds = AnswersDataset(
        test_df, 
        tok, 
        text_col="ResponseText.x",
        model_col="Model_Answer",
        question_col="Question",
        #rubric_col="Rubric",
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
    rows_for_csv = []

    with torch.inference_mode():
        for batch in test_loader:
            batch_idx = batch["idx"].cpu().numpy()

            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                global_attention_mask=batch.get("global_attention_mask", None),
            )

            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            preds_np = preds.cpu().numpy()
            labels_np = batch["labels"].cpu().numpy()
            conf_np = conf.cpu().numpy()

            for i, idx in enumerate(batch_idx):
                rows_for_csv.append({
                    "UNIV": test_df.iloc[idx]["UNIV"],
                    "Question_type": test_df.iloc[idx]["TaskPrompt"],
                    "true_label": int(labels_np[i]),
                    "pred_label": int(preds_np[i]),
                    "confidence": float(conf_np[i]),
                })

    test_acc = correct / total if total else 0.0
    print(f"\nFinal TEST accuracy: {test_acc:.4f}")

    preds_out_path = f"test_predictions_q_m.csv"
    pd.DataFrame(rows_for_csv).to_csv(preds_out_path, index=False)
    print("Saved predictions to:", os.path.abspath(preds_out_path))

    wandb.log({"test_acc": test_acc})
    wandb.save(preds_out_path)



if __name__ == "__main__":
    main()

