import os

import torch
from src.nlu.extractors.model import XLMR, mBERT, mDAPT, BioBERT, HnBERTvn, phoBERT, OUR
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    BertConfig
)

import matplotlib.pyplot as plt
import itertools
import pandas as pd

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMR, XLMRobertaTokenizer),
    "mbert": (BertConfig, mBERT, AutoTokenizer),
    "mdapt": (BertConfig, mDAPT, AutoTokenizer),
    "biobert": (BertConfig, BioBERT, AutoTokenizer),
    "hnbertvn": (RobertaConfig, HnBERTvn, AutoTokenizer),
    "phobert": (RobertaConfig, phoBERT, AutoTokenizer),
    "our": (BertConfig, OUR, AutoTokenizer),
    "our_v2": (BertConfig, OUR, AutoTokenizer),
    "our_v2_con": (BertConfig, OUR, AutoTokenizer),
    "pretrained": (BertConfig, OUR, AutoTokenizer),
    "pretrained_vn": (BertConfig, HnBERTvn, AutoTokenizer),
}

MODEL_PATH_MAP = {
    "xlmr": "/workspace/vinbrain/vutran/Backbone/XLMr/",
    "mbert": "/workspace/vinbrain/vutran/Backbone/mBERT/",
    "mdapt": "/workspace/vinbrain/vutran/Backbone/mDAPT/",
    "biobert": "/workspace/vinbrain/vutran/Backbone/BioBERT/",
    "hnbertvn": "/workspace/vinbrain/vutran/Backbone/HnBERTvn/",
    "phobert": "/workspace/vinbrain/vutran/Backbone/phoBERT/",
    "our": "/workspace/vinbrain/vutran/Transfer_Learning/Domain_Adaptive/Pretrain/src/mDAPT_vn_eng/",
    "our_v2": "/workspace/vinbrain/vutran/Transfer_Learning/Domain_Adaptive/Pretrain/src/mDAPT_vn_eng_v2/",
    "our_v2_con": "/workspace/vinbrain/vutran/Transfer_Learning/Domain_Adaptive/Pretrain/src/mDAPT_vn_eng_v2_continue",
    "pretrained": "/workspace/vinbrain/minhnp/KB/Pretraining/key_word/src/mDAPT_pretrained/",
    "pretrained_vn": "/workspace/vinbrain/minhnp/KB/Pretraining/key_word/src/HnBERT_pretrained/"
}



def get_intent_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.intent_label_file), "r", encoding="utf-8")
    ]

def get_slot_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.slot_label_file), "r", encoding="utf-8")
    ]

def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_dir)
     
def get_args(path_args):
    return torch.load(os.path.join(path_args, 'training_args.bin'))