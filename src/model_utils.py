import torch
import torch.nn as nn
from transformers import LongformerModel

class CustomLongformerClassifier(nn.Module):
    def __init__(self, model_name="allenai/longformer-base-4096", num_labels=3, dropout=0.1):
        super().__init__()
        self.base = LongformerModel.from_pretrained(model_name)
        hidden = self.base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, global_attention_mask=None, output_hidden_states=False):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        cls = outputs.last_hidden_state[:, 0, :]         
        logits = self.classifier(self.dropout(cls))     
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        if output_hidden_states:
            return {"loss": loss, "logits": logits, "hidden_states": outputs.hidden_states}
        return {"loss": loss, "logits": logits}

def build_model(model_name="allenai/longformer-base-4096", num_labels=3):
    return CustomLongformerClassifier(model_name=model_name, num_labels=num_labels)
