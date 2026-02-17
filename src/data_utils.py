from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import LongformerTokenizerFast
from torch.utils.data import DataLoader, Dataset
import torch


class AnswersDataset(Dataset):
    """
    Dataset that can either:
    - 'concat': concatenate question, student answer, model answer into one string
                and tokenize once.
    - 'sum':   tokenize question, student answer, model answer separately and
                later SUM their embeddings inside the model.
    """
    def __init__(
        self,
        df,
        tokenizer,
        text_col: str = "student_answer",
        model_col: str = "model_answer",
        question_col: str = "question",
        label_col: str = "label",
        max_len: int = 2048,
        fusion_mode: str = "concat",  # "concat" or "sum"
    ):
        assert fusion_mode in ("concat", "sum"), "fusion_mode must be 'concat' or 'sum'"

        self.texts = df[text_col].astype(str).fillna("").tolist()
        self.modelanswer = df[model_col].astype(str).fillna("").tolist()
        self.question = df[question_col].astype(str).fillna("").tolist()
        self.labels = df[label_col].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fusion_mode = fusion_mode

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        student_text = self.texts[idx]
        model_text = self.modelanswer[idx]
        question_text = self.question[idx]
        label = self.labels[idx]

        if self.fusion_mode == "concat":
            # ===== ORIGINAL CONCAT BEHAVIOR =====
            sep = self.tokenizer.sep_token
            combined = (
                f"Question: {question_text} {sep} "
                f"Student answer: {student_text} {sep} "
                f"Model answer: {model_text}"
            )

            enc = self.tokenizer(
                combined,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(label, dtype=torch.long)
            item["idx"] = torch.tensor(idx, dtype=torch.long)
            return item

        else:
            # ===== SUM MODE: tokenize three sequences separately =====
            enc_q = self.tokenizer(
                question_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc_s = self.tokenizer(
                student_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc_m = self.tokenizer(
                model_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )

            item = {
                "q_input_ids": enc_q["input_ids"].squeeze(0),
                "q_attention_mask": enc_q["attention_mask"].squeeze(0),
                "s_input_ids": enc_s["input_ids"].squeeze(0),
                "s_attention_mask": enc_s["attention_mask"].squeeze(0),
                "m_input_ids": enc_m["input_ids"].squeeze(0),
                "m_attention_mask": enc_m["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long),
            }
            return item


if __name__ == "__main__":
    # Quick sanity check
    df = pd.read_csv("data/CLASSIFIES_datatable.csv")
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    # Test concat mode
    train_dataset_concat = AnswersDataset(train_df, tokenizer, fusion_mode="concat")
    train_loader_concat = DataLoader(train_dataset_concat, batch_size=2, shuffle=True)

    # Test sum mode
    train_dataset_sum = AnswersDataset(train_df, tokenizer, fusion_mode="sum")
    train_loader_sum = DataLoader(train_dataset_sum, batch_size=2, shuffle=True)

    print("Concat mode batches:", len(train_loader_concat))
    print("Sum mode batches:", len(train_loader_sum))

