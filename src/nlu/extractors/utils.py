import logging
import os
import random

import numpy as np
# from sklearn.metrics import f1_score
import torch
from src.nlu.extractors.model import XLMR_NER, mBERT_NER, mDAPT_NER, BioBERT_NER, HnBERTvn_NER, phoBERT_NER, OUR_NER
from seqeval.metrics import precision_score, recall_score, classification_report, f1_score
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


def get_intent_labels(path):
    return [
        label.strip()
        for label in open(os.path.join(path), "r", encoding="utf-8")
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
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_dir)


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

def read_line_by_line(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())
    return data

def convert_label_one_hot(out_intent, label_intent_map):
    labels = []
    for i in out_intent:
        label = [0]*len(label_intent_map)
        i = i.split('\t')
        for j in i:
            j = label_intent_map[j]
            label[j] = 1
        labels.append(label)
    return labels

def get_attribute(preds):
    preds_intent = []
    preds_disease = []
    for i in preds:
        pred_disease = []
        pred_intent = []
        for j in i:
            if 'disease' not in j:
                pred_disease.append('O')
            else:
                pred_disease.append(j)
            if j != 'O' and j != 'UNK' and 'disease' not in j:
                pred_intent.append(j.split('-')[-1])
        if pred_intent:
            pred_intent = np.unique(pred_intent).tolist()
        else:
            pred_intent = ['overview']
        preds_intent.append("\t".join(pred_intent))
        preds_disease.append(pred_disease)
    return preds_disease, preds_intent

def get_result_attribute(preds_one_hot, labels_one_hot):
    from sklearn.metrics import classification_report, f1_score
    print(classification_report(labels_one_hot, preds_one_hot, digits=4))
    return f1_score(labels_one_hot, preds_one_hot,average='micro')

def get_slot_metrics(preds, labels):
    intent_label = get_intent_labels('../data/intent_label.txt')
    out_intent = read_line_by_line('../data/label.txt')
    assert len(preds) == len(labels) == len(out_intent)
    label_intent_map = {label: i for i, label in enumerate(intent_label)}
    labels_one_hot = convert_label_one_hot(out_intent, label_intent_map)
    preds, preds_intent = get_attribute(preds)
    write_txt_file('./predictions.txt', preds_intent)
    preds_one_hot = convert_label_one_hot(preds_intent, label_intent_map)
#     print(preds_intent)
    
    slot_f1_attr = get_result_attribute(preds_one_hot, labels_one_hot)
    conf_mat_dict = get_confusion_matrix(labels_one_hot, preds_one_hot)
    fig = plt.figure(figsize=(16, 9))
    columns = 4
    rows = 2
    count = 1
    for label, matrix in conf_mat_dict.items():
        label = ['not'+label, label]
#         print(matrix)
        plt.subplot(rows, columns, count)
        count += 1
        plot_confusion_matrix(matrix, label)
    fig.savefig('cm.png')
    
    print(classification_report(labels, preds, digits=4))
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
        "slot_f1_attr": slot_f1_attr,
        "mean_slot": 0.5*(f1_score(labels, preds) + slot_f1_attr)
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

def get_confusion_matrix(labels_one_hot, preds_one_hot):
    intent_label = get_intent_labels('../data/intent_label.txt')
    from sklearn.metrics import multilabel_confusion_matrix
    conf_mat_dict={}

#     for label_col in range(len(intent_label)):
#         y_true_label = labels_one_hot[:, label_col]
#         y_pred_label = preds_one_hot[:, label_col]
#         conf_mat_dict[intent_label[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)
#     print("LABEL")
#     print(labels_one_hot)
#     print("PRED")
#     print(preds_one_hot)
    cms = multilabel_confusion_matrix(np.array(labels_one_hot), np.array(preds_one_hot))
#     print(cms)
    for label, cm in zip(intent_label, cms):
        conf_mat_dict[label] = cm
    for label, matrix in conf_mat_dict.items():
        print("Confusion matrix for label {}:".format(label))
        print(matrix)
    return conf_mat_dict
        
        

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(classes[1])
#     plt.colorbar()
#     print(cm.shape)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#     name_image = classes[1]+'.png'
#     plt.savefig(name_image)
#     plt.remove()
    
# def plot_confusion_matrix(cm, classes):
#     df_cm = pd.DataFrame(cm, range(len(classes)), range(len(classes)))
#     # plt.figure(figsize=(10,7))
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#     name_image = classes[0]+'.png'
#     plt.savefig(name_image)
def write_txt_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write('%s\n' %item)
            

            
            
def get_args(path_args):
    return torch.load(os.path.join(path_args, 'training_args.bin'))