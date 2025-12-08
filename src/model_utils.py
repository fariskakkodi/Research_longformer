import torch
import torch.nn as nn
from transformers import LongformerModel


class CustomLongformerClassifier(nn.Module):
    """
    Custom Longformer classifier that supports two fusion modes:

    - 'concat': standard single-sequence input_ids/attention_mask
    - 'sum':    SUM embeddings of Question, Student Answer, Model Answer
    """
    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        num_labels: int = 3,
        dropout: float = 0.1,
        fusion_mode: str = "concat",  # "concat" or "sum"
    ):
        super().__init__()
        assert fusion_mode in ("concat", "sum"), "fusion_mode must be 'concat' or 'sum'"

        self.base = LongformerModel.from_pretrained(model_name)
        hidden = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.fusion_mode = fusion_mode

    def forward(
        self,
        # concat mode inputs
        input_ids=None,
        attention_mask=None,
        # sum mode inputs
        q_input_ids=None,
        s_input_ids=None,
        m_input_ids=None,
        q_attention_mask=None,
        s_attention_mask=None,
        m_attention_mask=None,
        # shared
        labels=None,
        global_attention_mask=None,
        output_hidden_states: bool = False,
    ):
        if self.fusion_mode == "concat":
            # ===== CONCAT MODE: behave like standard Longformer classifier =====
            outputs = self.base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            cls = outputs.last_hidden_state[:, 0, :]  # (B, H)
            logits = self.classifier(self.dropout(cls))

        else:
            # ===== SUM MODE: SUM embeddings of Q, S, M =====
            # 1) Get embeddings for each part
            emb_q = self.base.embeddings(q_input_ids)  # (B, L, H)
            emb_s = self.base.embeddings(s_input_ids)  # (B, L, H)
            emb_m = self.base.embeddings(m_input_ids)  # (B, L, H)

            # 2) Sum embeddings elementwise
            combined_embeds = emb_q + emb_s + emb_m  # (B, L, H)

            # 3) Combine attention masks with logical OR
            combined_attention_mask = (
                (q_attention_mask + s_attention_mask + m_attention_mask) > 0
            ).long()  # (B, L)

            # 4) If no global_attention_mask provided, make first token global
            if global_attention_mask is None:
                global_attention_mask = torch.zeros_like(combined_attention_mask)
                global_attention_mask[:, 0] = 1

            # 5) Run Longformer using inputs_embeds
            outputs = self.base(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                global_attention_mask=global_attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            cls = outputs.last_hidden_state[:, 0, :]  # (B, H)
            logits = self.classifier(self.dropout(cls))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        if output_hidden_states:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states,
            }
        return {"loss": loss, "logits": logits}


def build_model(
    model_name: str = "allenai/longformer-base-4096",
    num_labels: int = 3,
    fusion_mode: str = "concat",
):
    return CustomLongformerClassifier(
        model_name=model_name,
        num_labels=num_labels,
        fusion_mode=fusion_mode,
    )

