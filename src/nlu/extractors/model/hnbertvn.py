import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
# # from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
# from transformers import BertPreTrainedModel, BertModel

from .module import SlotClassifier, Classifier


class HnBERTvn(RobertaPreTrainedModel):
    def __init__(self, config, args, slot_label_lst, label_lst):
        super(HnBERTvn, self).__init__(config)
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        self.num_labels = len(label_lst)
        self.roberta = RobertaModel(config)  # Load pretrained bert

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_slot_labels,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        self.classifier = Classifier(
            config.hidden_size,
            self.num_labels,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids=None, slot_labels_ids=None, labels_ids=None):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        
        slot_logits = self.slot_classifier(sequence_output)
        logits = self.classifier(pooled_output)

        total_loss = 0

        if labels_ids is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels_ids.view(-1))
            loss = loss_fct(logits, labels_ids)
            total_loss += self.args.intent_loss_coef * loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction="mean")
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += (1 - self.args.intent_loss_coef) * slot_loss

        outputs = ((logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits