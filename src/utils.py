import logging
import os
import random

import numpy as np
import torch
from model import XLMR_NER, mBERT_NER, mDAPT_NER, BioBERT_NER, HnBERTvn_NER, phoBERT_NER, OUR_NER
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    BertConfig
)


MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMR_NER, XLMRobertaTokenizer),
    "mbert": (BertConfig, mBERT_NER, AutoTokenizer),
    "mdapt": (BertConfig, mDAPT_NER, AutoTokenizer),
    "biobert": (BertConfig, BioBERT_NER, AutoTokenizer),
    "hnbertvn": (RobertaConfig, HnBERTvn_NER, AutoTokenizer),
    "phobert": (RobertaConfig, phoBERT_NER, AutoTokenizer),
    "our": (BertConfig, OUR_NER, AutoTokenizer),
    "our_v2": (BertConfig, OUR_NER, AutoTokenizer),
    "our_v2_con": (BertConfig, OUR_NER, AutoTokenizer)
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
    "our_v2_con": "/workspace/vinbrain/vutran/Transfer_Learning/Domain_Adaptive/Pretrain/src/mDAPT_vn_eng_v2_continue"
}


def get_intent_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.token_level, args.intent_label_file), "r", encoding="utf-8")
    ]


# def get_slot_labels(args):
#     return [
#         label.strip()
#         for label in open(os.path.join(args.data_dir, args.token_level, args.slot_label_file), "r", encoding="utf-8")
#     ]

def get_slot_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.data_dir, args.slot_label_file), "r", encoding="utf-8")
    ]

def load_tokenizer(args):
#     return MODEL_CLASSES[args.model_type][2].from_pretrained("/workspace/vinbrain/vutran/Backbone/BioBERT")
#     return MODEL_CLASSES[args.model_type][2].from_pretrained("/workspace/vinbrain/minhnp/pretrainedLM/phobert-base")
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    results.update(slot_result)
    return results



def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    print(classification_report(labels, preds, digits=4))
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {"intent_acc": acc}


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8")]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = intent_preds == intent_labels

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return {"semantic_frame_acc": semantic_acc}
