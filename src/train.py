import torch
import pandas as pd
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
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tok = LongformerTokenizerFast.from_pretrained(model_name)
    train_ds = AnswersDataset(train_df, tok, max_len=max_len)
    val_ds   = AnswersDataset(val_df, tok, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False)

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
        with torch.no_grad():
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

if __name__ == "__main__":
    main()
