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
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging

@dataclass
class MultimodalInputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.

        image_path:  (Optional) string. the image path of the multimodal post.

        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    image_path: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

class MultimodalClassificationProcessor(DataProcessor):
    """
    Data processor for MVSA multimoal(text+image) classification datasets (MVSA-S, MVSA-M).
    """

    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return MultimodalInputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            tensor_dict["image_path"].numpy().decode("utf-8"), 
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
        file_name = os.path.join(data_dir, 'val.json')
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
            image_path = line['image']
            label = line['label']
            examples.append(MultimodalInputExample(guid=guid, text_a=text_a, image_path=image_path, label=label))
        return examples


class TumblrMultimodalClassificationProcessor(DataProcessor):
    """
    Data processor for MVSA multimoal(text+image) classification datasets (MVSA-S, MVSA-M).
    """

    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return MultimodalInputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            tensor_dict["image_path"].numpy().decode("utf-8"), 
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
        file_name = os.path.join(data_dir, 'val.json')
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
            image_path = line['image']
            label = line['label']
            examples.append(MultimodalInputExample(guid=guid, text_a=text_a, image_path=image_path, label=label))
        return examples


