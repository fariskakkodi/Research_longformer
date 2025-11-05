import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizerFast

from src.data_utils import AnswersDataset
from src.model_utils import build_model

def main():
    csv_path = "data/PSU_q2_dataset.csv"
    model_name = "allenai/longformer-base-4096"
    max_len = 512
    train_bs = 2
    val_bs = 2
    epochs = 5
    lr = 2e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path).dropna(subset=["student_answer", "label"]).copy()
    df["label"] = df["label"].astype(int)

    # Train/Val/Test: 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    tok = LongformerTokenizerFast.from_pretrained(model_name)
    train_ds = AnswersDataset(train_df, tok, max_len=max_len)
    val_ds   = AnswersDataset(val_df, tok, max_len=max_len)
    test_ds  = AnswersDataset(test_df, tok, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=val_bs,   shuffle=False)

    model = build_model(model_name, num_labels=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
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

        model.eval()
        total = correct = 0
        with torch.inference_mode():
            for batch in val_loader:
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
        acc = correct / total if total else 0.0
        print(f"epoch {epoch}  val_acc={acc:.4f}")

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
        print("Correlation between predictions and labels: undefined (zero variance)")
    else:
        corr = np.corrcoef(all_preds, all_labels)[0, 1]
        print(f"Correlation between predictions and labels: {corr:.4f}")

if __name__ == "__main__":
    main()

