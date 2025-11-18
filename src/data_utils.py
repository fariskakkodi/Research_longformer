from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import LongformerTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch

class AnswersDataset(Dataset):
    def __init__(self, df, tokenizer, text_col="student_answer", model_col="model_answer", label_col="label", max_len=2048):
        self.texts = df[text_col].astype(str).fillna("").tolist()
        self.modelanswer = df[model_col].astype(str).fillna("").tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        student_text = self.texts[idx]
        model_text = self.texts[idx]
        label = self.labels[idx]
        combined = (
            "Student answer: " + student_text + "\n"
            "Model answer: " + model_text
        )
        enc = self.tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


if __name__ == "__main__":
    df = pd.read_csv("data/CLASSIFIES_datatable.csv")
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    train_dataset = AnswersDataset(train_df, tokenizer)
    val_dataset = AnswersDataset(val_df, tokenizer)
    test_dataset = AnswersDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(len(train_loader), len(val_loader), len(test_loader))
