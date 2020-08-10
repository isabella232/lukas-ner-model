from datetime import date

from transformers import BertForSequenceClassification, BertModel, BertLayer

import torch
from torch.nn import Dropout, Linear, BCEWithLogitsLoss


class BertForMultiLabelSequenceClassification(BertForSequenceClassification):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
        `num_parent_labels`: the number of classes for the parent classifier. Default = 0.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2, num_labels_parent=0):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(
            config.hidden_size + num_labels_parent, config.num_labels
        )
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        parent_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        if parent_labels is not None:
            pooled_output = torch.cat((pooled_output, parent_labels), 1)
            
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_embeddings(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_bert_embeddings(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True

    def freeze_bert_encoder(self):
        for param in self.bert.encoder.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.encoder.parameters():
            param.requires_grad = True

    def save(self):
        """Saves the model as a binary file."""
        model_to_save = (
            self.module if hasattr(self, "module") else self
        )  # Only save the model itself
        d = date.today().strftime("%Y-%m-%d")
        output_model_file = f"mltc/data/model_files/finetuned_{d}_pytorch_model.bin"
        torch.save(model_to_save.state_dict(), output_model_file)


class BertForMultilabelSpecialized(BertForMultiLabelSequenceClassification):
    """A subclass that freezes the layers from a pretrained classifier whilst adding
    an additional transformer block that is finetuned during training.
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultilabelSpecialized, self).__init__(config)
        self.bert = BertModel(config)
        self.freeze_bert_embeddings()
        self.freeze_bert_encoder()
        self.bert.encoder.add_module("12", BertLayer(config))
        self.apply(self._init_weights)

