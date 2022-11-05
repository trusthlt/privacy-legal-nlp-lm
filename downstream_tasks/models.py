import torch
import torch.nn as nn
from transformers import AutoModel


class BERTForMultiLabelClassification(torch.nn.Module):
    def __init__(self, pretrained_weights='bert-base-uncased', from_flax=False, from_tf=False, num_labels=2):
        super(BERTForMultiLabelClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(pretrained_weights, from_flax=from_flax, from_tf=from_tf,)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, class_weights=None):
        output_1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.dropout(output_1[1])
        logits = self.classifier(output_2)
        output = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss(
                    reduction='mean',
                    pos_weight=class_weights,
                )
                loss = loss_fct(logits, labels)
            output = (loss,) + output

        return output


class BERTForSequenceClassification(torch.nn.Module):
    def __init__(self, pretrained_weights='bert-base-uncased', from_flax=False, from_tf=False, num_labels=2):
        super(BERTForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(pretrained_weights, from_flax=from_flax, from_tf=from_tf,)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, class_weights=None):
        output_1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output_2 = self.dropout(output_1[1])
        logits = self.classifier(output_2)
        output = (logits,)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = (loss,) + output
        return output