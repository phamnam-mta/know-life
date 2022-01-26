import re

import torch
import torch.nn as nn
import numpy as np

from src.nlu.extractors.utils import MODEL_CLASSES, get_args, load_tokenizer, get_slot_labels, get_intent_labels
from seqeval.metrics.sequence_labeling import get_entities


class BERTEntityExtractor():
    def __init__(self, model_dir, data_dir = None):
        
        self.args = get_args(model_dir)
  
        self.args.data_dir = data_dir
        self.args.model_dir = model_dir

        self.intent_label_lst = get_intent_labels(self.args)
        self.intent_label_map = {i: label for i, label in enumerate(self.intent_label_lst)}

        self.slot_label_lst = get_slot_labels(self.args)
        self.slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}

        self.tokenizer = load_tokenizer(self.args)
        print('****loaded tokenizer****')
        
        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type]
        self.config = self.config_class.from_pretrained(self.args.model_dir, finetuning_task="syllable")
        
        self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                config=self.config,
                                                args=self.args,
                                                slot_label_lst=self.slot_label_lst,
                                                label_lst=self.intent_label_lst
                                                )
        print("***** Model Loaded *****")

        self.sigmoid_fct = nn.Sigmoid()
        
        self.device = 'cpu'

        self.model.to(self.device)
        self.model.eval()
        
    def inference(self, text):
        text = self.padding_punct(text)
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
#         print(outputs)

        intent_logit = outputs[0]
#         print(intent_logit)
        intent_list = self.convert_logit_to_intent(intent_logit)

        slot_logit = outputs[1]
        slot_list = self.convert_logit_to_entity(slot_logit)

        predictions = self.prediction(slot_list, tokens, attention_mask, intent_list)
        return predictions

        
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

    def convert_logit_to_intent(self, outputs):
        outputs = self.sigmoid_fct(outputs).detach().cpu().numpy()
        outputs = np.where(outputs > self.args.threshold, 1, 0)
#         print(outputs)
        intent_list = [[] for _ in range(outputs.shape[0])]
        for i in range(outputs.shape[0]):
            for idx, j in enumerate(range(outputs.shape[1])):
                if outputs[i][j] == 1:
                    intent_list[i].append(self.intent_label_map[idx])
        return intent_list
    
    def convert_logit_to_entity(self, outputs):
        outputs = outputs.detach().cpu().numpy()
        outputs = np.argmax(outputs, axis=2)
        slot_list = [[] for _ in range(outputs.shape[0])]
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
#                 if j == 0: continue
                slot_list[i].append(self.slot_label_map[outputs[i][j]])
        return slot_list
    
    def prediction(self, slot_preds, tokens, attention_mask, intent_list):
        predictions = {
            'disease': [],
            'symptom': [],
            'intent': [],
        }
        seq_in = []
        seq_out = []
        for s, t, a in zip(slot_preds[0][1:], tokens[1:], attention_mask[0][1:]):
            if a == 1:
                seq_in.append(t)
                seq_out.append(s)
        seq_in = seq_in[:-1]
        seq_out = seq_out[:-1]
#         print(seq_in)
#         print(seq_out)
        entitys = get_entities(seq_out)
#         print(entitys)
        for entity in entitys:
            value = seq_in[entity[1]: entity[2]+1]
            value = " ".join(value)
            value = self.post_process_disease(value)
            predictions[entity[0].lower()].append(value)
        predictions["intent"] = intent_list[0]
        return predictions
    def post_process_disease(self, text):
        text = text.replace('@@ ', '')
        text = text.replace(' ##', '')
        return text

    def padding_punct(self, s):
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        s = s.strip().lower()
        return s

        