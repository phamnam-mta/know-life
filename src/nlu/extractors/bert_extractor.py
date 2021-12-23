import torch
import numpy as np

from src.nlu.extractors.utils import MODEL_CLASSES, get_args, load_tokenizer, get_slot_labels
from seqeval.metrics.sequence_labeling import get_entities


class BERTEntityExtractor():
    def __init__(self, model_dir, data_dir = None):
        
        self.args = get_args(model_dir)
  
        self.args.data_dir = data_dir
        self.args.model_dir = model_dir
        self.slot_label_lst = get_slot_labels(self.args)
        self.slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        # print(self.args.model_dir)
        self.tokenizer = load_tokenizer(self.args)
        print('****loaded tokenizer****')
        # print(self.slot_label_map)
        # print(glob.glob(path_args + '/*'))
        
        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type]
        self.config = self.config_class.from_pretrained(self.args.model_dir, finetuning_task="syllable")
        # print(self.config)
        

        self.model = self.model_class.from_pretrained(model_dir, config=self.config, args=self.args, slot_label_lst=self.slot_label_lst)
        print("***** Model Loaded *****")
        
        self.device = 'cpu'

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
        # print(slot_list)
        predictions = self.get_disease_attribute(slot_list, tokens, attention_mask)
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
    
    def convert_logit_to_entity(self, outputs):
        outputs = outputs.detach().cpu().numpy()
        outputs = np.argmax(outputs, axis=2)
        slot_list = [[] for _ in range(outputs.shape[0])]
        for i in range(outputs.shape[0]):
            for j in range(outputs.shape[1]):
#                 if j == 0: continue
                slot_list[i].append(self.slot_label_map[outputs[i][j]])
        return slot_list
    
    def get_disease_attribute(self, slot_preds, tokens, attention_mask):
        predictions = []
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
            predictions.append({
                'key': entity[0],
                'value': value
            })
        # print(predictions)
        if len(predictions) == 1 and predictions[0]['key'] == 'disease':
            predictions.append({
                'key': 'overview',
                'value': ''
            })

        return predictions
    def post_process_disease(self, text):
        text = text.replace('@@ ', '')
        return text

        