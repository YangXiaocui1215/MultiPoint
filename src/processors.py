"""Dataset utils for different data settings for GLUE."""

from email.mime import image
import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from src.multimodal_processors import MultimodalInputExample, MultimodalClassificationProcessor, TumblrMultimodalClassificationProcessor
from src.multimodal_processors_add_caption import AddCaptionMVSAClassificationProcessor, AddCaptionTumblrMultimodalClassificationProcessor
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging
from sklearn import metrics


logger = logging.getLogger(__name__)

class MVSAClassificationProcessor(DataProcessor):
    """
    Data processor for MVSA text classification datasets (MVSA-S, MVSA-M).
    """

    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        ) 
    def get_train_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'train.json')
        train_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                train_dataset.append(json_line)
        return self._create_examples(train_dataset, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'dev.json')
        dev_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                dev_dataset.append(json_line)
        return self._create_examples(dev_dataset, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'test.json')
        test_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                test_dataset.append(json_line)
        return self._create_examples(test_dataset, "test")
    
    def get_labels(self):
        """See base class."""
        return ["negative", "neutral", "positive"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line['id'])
            text_a = line['text']
            label = line['label']
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class TumblrClassificationProcessor(DataProcessor):
    """
    Data processor for MVSA text classification datasets (MVSA-S, MVSA-M).
    """

    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        ) 
    def get_train_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'train.json')
        train_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                train_dataset.append(json_line)
        return self._create_examples(train_dataset, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'dev.json')
        dev_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                dev_dataset.append(json_line)
        return self._create_examples(dev_dataset, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        file_name = os.path.join(data_dir, 'test.json')
        test_dataset=[]
        with open(file_name, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                # print(json_line)
                test_dataset.append(json_line)
        return self._create_examples(test_dataset, "test")
    
    def get_labels(self):
        """See base class."""
        return ["angry", "bored", "calm", "fear", "happy", "love", "sad"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line['id'])
            text_a = line['text']
            label = line['label']
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

class TextClassificationProcessor(DataProcessor):
    """
    Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa).
    """

    def __init__(self, task_name):
        self.task_name = task_name 

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(InputExample(guid=guid, text_a=line[1] + '. ' + line[2], short_text=line[1] + ".", label=line[0]))
            elif self.task_name == "yelp_review_full":
                examples.append(InputExample(guid=guid, text_a=line[1], short_text=line[1], label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += ' ' + line[2]
                if not pd.isna(line[3]):
                    text += ' ' + line[3]
                examples.append(InputExample(guid=guid, text_a=text, short_text=line[1], label=line[0])) 
            elif self.task_name in ['mr', 'sst-5', 'subj', 'trec', 'cr', 'mpqa']:
                examples.append(InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples
        
def text_classification_metrics(task_name, preds, labels):
    if task_name == 'mvsa_single' or task_name =='mvsa_multiple':
        label_list = ["negative", "neutral", "positive"]
        labels_list = [i for i in range(len(label_list))]
        report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)
        return {"acc": (preds == labels).mean(),
                "report": report}
    else:
        return {"acc": (preds == labels).mean()}

def multimodal_classification_metrics(task_name, preds, labels):
  
    label_list = ["negative", "neutral", "positive"]
    labels_list = [i for i in range(len(label_list))]
    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)
    return {"acc": (preds == labels).mean(),
                "report": report}

def tumblr_classification_metrics(task_name, preds, labels):
    
    label_list = ["angry", "bored", "calm", "fear", "happy", "love", "sad"]
    labels_list = [i for i in range(len(label_list))]
    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)
    return {"acc": (preds == labels).mean(),
            "report": report}

# Add your task to the following mappings

processors_mapping = {
    "mul_mvsa_single_fusion_add_caption": AddCaptionMVSAClassificationProcessor('mul_mvsa_single_fusion_add_caption'),
    "mul_mvsa_multiple_fusion_add_caption": AddCaptionMVSAClassificationProcessor('mul_mvsa_multiple_fusion_add_caption'),
    "mul_tumblr_fusion_add_caption": AddCaptionTumblrMultimodalClassificationProcessor('mul_tumblr_fusion_add_caption'),
}

num_labels_mapping = {
    "mul_mvsa_single_fusion_add_caption": 3,

    "mul_mvsa_multiple_fusion_add_caption": 3,

    "mul_tumbl_fusion_add_caption": 7,
}

output_modes_mapping = {
    "mul_mvsa_single_fusion_add_caption":"classification",
    "mul_mvsa_multiple_fusion_add_caption": "classification",
    "mul_tumbl_fusion_add_caption": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "mul_mvsa_single_fusion_add_caption": multimodal_classification_metrics,
    "mul_mvsa_multiple_fusion_add_caption": multimodal_classification_metrics,
    "mul_tumbl_fusion_add_caption": tumblr_classification_metrics,
}

median_mapping = {
    "sts-b": 2.5
}

bound_mapping = {
    "sts-b": (0, 5)
}