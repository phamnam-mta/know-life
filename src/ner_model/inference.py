from src.ner_model.model import (
    XLMR_NER,
    mBERT_NER,
    mDAPT_NER,
    BioBERT_NER,
    HnBERTvn_NER,
    phoBERT_NER,
    OUR_NER
)
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    BertConfig
)
import os
import torch
import numpy as np

from src.ner_model.utils import read_line_by_line, MODEL_CLASSES, get_args, load_tokenizer, get_slot_labels


class Inference:
    def __init__(self,path_args, data_dir = './data'):
        
        self.args = get_args(path_args)
  
        self.args.data_dir = data_dir
        self.slot_label_lst = get_slot_labels(self.args)
        self.slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        
        self.tokenizer = load_tokenizer(self.args)
        
        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type]
        self.config = self.config_class.from_pretrained(self.args.model_dir, finetuning_task="syllable")

        self.model = self.model_class.from_pretrained(path_args, config=self.config, args=self.args, slot_label_lst=self.slot_label_lst)
        print("***** Model Loaded *****")
        
        self.device = self.args.device

        self.model.to(self.device)
        self.model.eval()
        
    def inference(self, text):
        input_ids, attention_mask, token_type_ids, tokens = self.process_data(text)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        
        inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
        if self.args.model_type != "distilbert":
            inputs["token_type_ids"] = token_type_ids
        outputs = self.model(**inputs)[1]
        slot_list = self.convert_logit_to_entity(outputs)
        print(slot_list)
        diseases, attributes = self.get_disease_attribute(slot_list, tokens)
        return {
            diseases[0][0]: attributes
        }
        
        
    def process_data(self, text):
         # Setting based on the current model type
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        unk_token = self.tokenizer.unk_token
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_label_id = self.args.ignore_index
        
        # Tokenize word by word (for NER)
        tokens = []
        for word in text.split():
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]
            tokens.extend(word_tokens)
                
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > self.args.max_seq_len - special_tokens_count:
            tokens = tokens[: (self.args.max_seq_len - special_tokens_count)]
        tokens += [sep_token]
        token_type_ids = [0] * len(tokens)
        
        tokens = [cls_token] + tokens
        token_type_ids = [0] + token_type_ids
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1 if True else 0] * len(input_ids)
        
        # Zero-pad up to the sequence length.
        padding_length = self.args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if True else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_label_id] * padding_length)
        
        assert len(input_ids) == self.args.max_seq_len, "Error with input length {} vs {}".format(len(input_ids), self.args.max_seq_len)
        assert len(attention_mask) == self.args.max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), self.args.max_seq_len
        )
        assert len(token_type_ids) == self.args.max_seq_len, "Error with token type length {} vs {}".format(
            len(token_type_ids), self.args.max_seq_len
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        return input_ids, attention_mask, token_type_ids, tokens
    
    def convert_logit_to_entity(self, outputs):
        outputs = outputs.detach().cpu().numpy()
        outputs = np.argmax(outputs, axis=2)
        slot_list = [[] for _ in range(outputs.shape[0])]
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
#                 if j == 0: continue
                slot_list[i].append(self.slot_label_map[outputs[i][j]])
        return slot_list
    
    def get_disease_attribute(self, slot_preds, tokens):
        preds_attr = []
        preds_disease = []
        for slot_pred in slot_preds:
            slot_disease = []
            pred_attr = []
            for value in slot_pred:
                if 'disease' not in value:
                    slot_disease.append('O')
                else:
                    slot_disease.append(value)
                if value != 'O' and value != 'UNK' and 'disease' not in value:
                    pred_attr.append(value.split('-')[-1])
            if pred_attr:
                pred_attr = np.unique(pred_attr).tolist()
            else:
                pred_attr = ['overview']
            diseases = []
            tmp = []
            for value, sub_token in zip(slot_disease, tokens):
                if 'B' in value and not tmp:
                    tmp.append(sub_token)
                elif 'B' in value and tmp:
                    print(tmp)
                    diseases.append(" ".join(tmp))
                    tmp = []
                elif 'I' in value:
                    tmp.append(sub_token)
                else:
                    if tmp:
                        print(tmp)
                        diseases.append(" ".join(tmp))
                        tmp = []
                    else:
                        continue
            if not diseases:
                diseases = ["not_disease"]
            preds_attr.append("\t".join(pred_attr))
            preds_disease.append(diseases)
        return preds_disease, preds_attr

        