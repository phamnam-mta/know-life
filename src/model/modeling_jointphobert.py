import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from torch import Tensor, device, dtype, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .module import IntentClassifier, SlotClassifier



class JointPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config, args, slot_label_lst):
        super(JointPhoBERT, self).__init__(config)
        self.args = args
        # self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        
        if self.args.num_layer_bert > 0 :
            self.roberta_embed, self.robert_layers, self.pooler = self.prunning_roberta(RobertaModel(config))
        else:
            self.roberta = RobertaModel(config)  # Load pretrained phobert

        self.use_intent_self_attn = args.use_intent_self_attn

        self.train_ner_only = args.train_ner_only
        if not args.train_ner_only:
            self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate,args.use_intent_self_attn)

        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_slot_labels,
            self.args.use_intent_context_concat,
            self.args.use_intent_context_attention,
            self.args.max_seq_len,
            self.args.attention_embedding_size,
            args.dropout_rate,
        )

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.use_tcn = args.use_tcn
        if args.use_tcn:
            num_channels = [args.nhid] * (args.levels - 1) + [config.hidden_size]

            self.tcn = TemporalConvNet(config.hidden_size, num_channels, args.kernel_size, dropout=args.dropout_tcn)
            self.embed_drop = nn.Dropout(0.1)
            self.decoder = nn.Linear(num_channels[-1], config.hidden_size)
        
    def prunning_roberta(self,roberta):
        roberta_embed = roberta.base_model.embeddings
        roberta_layers = nn.ModuleList([roberta.base_model.encoder.layer[i].to(self.args.device) for i in range(self.args.num_layer_bert,12,1)])
        pooler = roberta.base_model.pooler
        return roberta_embed,roberta_layers, pooler

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        if self.args.num_layer_bert > 0:
            outputs = self.roberta_embed(input_ids)
            ext_attention_mask = get_extended_attention_mask(attention_mask,input_ids.shape,input_ids.device)
            
            for layer in self.robert_layers:
                outputs = layer(outputs, attention_mask=ext_attention_mask)[0]

            if self.use_tcn:
                emb = self.embed_drop(outputs)
                tcn_out = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
                sequence_output = self.decoder(tcn_out)
            else:
                sequence_output = outputs
            
            pooled_output = self.pooler(outputs)  # [CLS]
            outputs = (outputs,)
        else:
            outputs = self.roberta(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )  # sequence_output, pooled_output, (hidden_states), (attentions)

            if self.use_tcn:
                emb = self.embed_drop(outputs[0])
                tcn_out = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
                sequence_output = self.decoder(tcn_out)
            else:
                sequence_output = outputs[0]
            pooled_output = outputs[1]  # [CLS]

        if not self.train_ner_only:
            if self.use_intent_self_attn:
                intent_logits = self.intent_classifier(sequence_output)
            else:
                intent_logits = self.intent_classifier(pooled_output)
        
        if not self.args.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask

        if self.args.embedding_type == "hard" and not self.train_ner_only:
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            slot_logits = self.slot_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            if self.train_ner_only:
                slot_logits = self.slot_classifier(sequence_output, None, tmp_attention_mask)
            else:
                slot_logits = self.slot_classifier(sequence_output, intent_logits, tmp_attention_mask)

        total_loss = 0
        # 1. Intent Softmax
        # if intent_label_ids is not None:
        #     if self.num_intent_labels == 1:
        #         intent_loss_fct = nn.MSELoss()
        #         intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
        #     else:
        #         intent_loss_fct = nn.CrossEntropyLoss()
        #         intent_loss = intent_loss_fct(
        #             intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
        #         )
        #     total_loss += self.args.intent_loss_coef * intent_loss

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
        
        if intent_label_ids is not None:
            if len(outputs) > 1:
                outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here
            else:
                outputs = ((intent_logits, slot_logits),)
            outputs = (total_loss,) + outputs

            return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
        else:
            if len(outputs) > 1:
                outputs = (slot_logits,) + outputs[2:]  # add hidden states and attention if they are here
            else:
                outputs = (slot_logits,)
            outputs = (total_loss,) + outputs
            return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
      """
      Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

      Arguments:
          attention_mask (:obj:`torch.Tensor`):
              Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
          input_shape (:obj:`Tuple[int]`):
              The shape of the input to the model.
          device: (:obj:`torch.device`):
              The device of the input to the model.

      Returns:
          :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
      """
      # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
      # ourselves in which case we just need to make it broadcastable to all heads.
      if attention_mask.dim() == 3:
          extended_attention_mask = attention_mask[:, None, :, :]
      elif attention_mask.dim() == 2:
          # Provided a padding mask of dimensions [batch_size, seq_length]
          # - if the model is a decoder, apply a causal mask in addition to the padding mask
          # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
          extended_attention_mask = attention_mask[:, None, None, :]
      else:
          raise ValueError(
              f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
          )

      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
      extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
      return extended_attention_mask