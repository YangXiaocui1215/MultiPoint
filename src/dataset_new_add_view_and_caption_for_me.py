"""Dataset utils for different data settings for GLUE."""

from lib2to3.pgen2.token import OP
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
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, median_mapping
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd

from PIL import Image, ImageFile, UnidentifiedImageError
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, Lambda
from transformers import GPT2Tokenizer, AutoFeatureExtractor, CLIPFeatureExtractor

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_TOKEN = "<image>"
NUM_IMAGE_TOKENS = 2
PAD_TOKEN_ID = 1

PROMPT_TOKEN="<prompt>"
SPECIAL_TOKEN_DICT = {'additional_special_tokens': [IMAGE_TOKEN, PROMPT_TOKEN]}

TIMM_CONFIGS = {
    'nf_resnet50':  {
        'input_size': (3, 256, 256),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 0.94,
    },
}

def is_clip_model(model_name):
    return model_name.startswith('openai/clip-')

def get_image_transform(model_name):
    if model_name in TIMM_CONFIGS.keys():
        config = TIMM_CONFIGS[model_name]
        transform = create_transform(**config)
        transform.transforms.append(
            Lambda(lambda x: x.unsqueeze(0)),
        )
    elif is_clip_model(model_name):
        transform = CLIPFeatureExtractor.from_pretrained(model_name)
    else:
        transform = AutoFeatureExtractor.from_pretrained(model_name)
    return transform

@dataclass(frozen=True)
class OurInputFeatures(InputFeatures):
    """
    Inherit from Transformers' InputFeatuers.
    """

    input_ids: List[int]  = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None # Position of the mask token
    label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    image_pixel_values_list: Optional[Optional[float]] = None
    raw_image: Optional[List[float]] = None
    image_token_mask: Optional[List[int]] = None

    view1_input_ids: List[int]  = None
    view1_attention_mask: Optional[List[int]] = None
    view1_token_type_ids: Optional[List[int]] = None
    view1_label: Optional[Union[int, float]] = None
    view1_mask_pos: Optional[List[int]] = None # Position of the mask token
    view1_label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    view1_image_pixel_values_list: Optional[Optional[float]] = None
    view1_raw_image: Optional[List[float]] = None
    view1_image_token_mask: Optional[List[int]] = None

    view2_input_ids: List[int] = None
    view2_attention_mask: Optional[List[int]] = None
    view2_token_type_ids: Optional[List[int]] = None
    view2_label: Optional[Union[int, float]] = None
    view2_mask_pos: Optional[List[int]] = None # Position of the mask token
    view2_label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    view2_image_pixel_values_list: Optional[Optional[float]] = None
    view2_raw_image: Optional[List[float]] = None
    view2_image_token_mask: Optional[List[int]] = None


    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def input_example_to_string(example, sep_token): 
    if example.text_b is None:
        return example.text_a
    else:
        # Warning: very simple hack here
        return example.text_a + ' ' + sep_token + ' ' + example.text_b

def input_example_to_tuple(example): 
    if example.text_b is None:
        if pd.isna(example.text_a) or example.text_a is None:
            return ['']
            logger.warn("Empty input")
        else:
            return [example.text_a]
    else:
        return [example.text_a, example.text_b]



def tokenize_multipart_input(
    input_text_list, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
):
    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None

        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 'mask': tokenizer.mask_token_id, 'sep': tokenizer.sep_token_id, 'sep+': tokenizer.sep_token_id, 
        }
        template_list = template.split('*') # Get variable list in the template
        
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            new_tokens = []
            segment_plus_1_flag = False
            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id]) 
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_tokens += enc(' ' + input_text_list[sent_id])
            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_tokens += enc(input_text_list[sent_id][:-1])
            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            if part[:4] == 'sent' or part[1:5] == 'sent':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]

            input_ids += new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]
            input_ids += input_tokens
            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__: # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    return result



def tokenize_multipart_input_add_image(
    input_text_list, 
    max_length, 
    tokenizer, 
    task_name=None, 
    prompt=False, 
    add_image=True,
    num_image_tokens=0,
    num_prompt_tokens=1,
    image_path_list=None,
    image_caption_list=None,
    image_transform=None,
    template=None,
    label_word_list=None, 
    first_sent_limit=None,
    other_sent_limit=None,
    gpt3=False,
    truncate_head=False,
    support_labels=None,
    use_demo=False
):  
    
    ###add <image> token to all_special_token
    if not IMAGE_TOKEN in tokenizer.all_special_tokens and not PROMPT_TOKEN in tokenizer.all_special_tokens:
        tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT)

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)


    def _read_image(image_path):
        raw = Image.open(image_path)
        raw = raw.convert('RGB') if raw.mode != 'RGB' else raw
        if isinstance(image_transform, Compose):
            image = image_transform(raw)
        elif image_transform is not None:  # HuggingFace
            image = image_transform(raw, return_tensors='pt')
            image = image['pixel_values']
        return raw, image
    
    def _add_image_tokens(text):
        N = num_image_tokens
        if N is not None or N > 0:
            tokens = ' '.join([IMAGE_TOKEN for x in range(N)])
            text = f'{tokens} {text}'
        return text
    
    def image_token_id():
        return tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    

    

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    image_pixel_values_list = [] ## for image pixel values

    N = num_image_tokens
    I = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    P = tokenizer.convert_tokens_to_ids(PROMPT_TOKEN)
    
    if prompt:
        """
        Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]
            *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
        """
        assert template is not None
   
        special_token_mapping = {
            'cls': tokenizer.cls_token_id, 
            'mask': tokenizer.mask_token_id, 
            'sep': tokenizer.sep_token_id, 
            'sep+': tokenizer.sep_token_id, 
            IMAGE_TOKEN: I,
            PROMPT_TOKEN: P
        }
        template_list = template.split('*') # Get variable list in the template
        # print('****************************template_list*********************************')
        # print('the template_list is {}'.format(template_list))
        '''
        ****************************template_list*********************************
        the template_list is ['', 'cls', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_0', '"_has', 'mask', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_1', '"_has', 'label_0', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_2', '"_has', 'label_1', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_3', '"_has', 'label_2', '_emotion', 'sep+', '']
        '''
        segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

        for part_id, part in enumerate(template_list):
            new_tokens = []
            new_image = None
            segment_plus_1_flag = False

            if part in special_token_mapping:
                if part == 'cls' and 'T5' in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                if part == PROMPT_TOKEN:
                    for p in range(num_prompt_tokens):
                        new_tokens.append(special_token_mapping[part])
                else:
                    new_tokens.append(special_token_mapping[part])
                if part == 'sep+':
                    segment_plus_1_flag = True
            elif part[:6] == 'label_':
                # Note that label_word_list already has extra space, so do not add more space ahead of it.
                label_id = int(part.split('_')[1])
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:7] == 'labelx_':
                instance_id = int(part.split('_')[1])
                label_id = support_labels[instance_id]
                label_word = label_word_list[label_id]
                new_tokens.append(label_word)
            elif part[:5] == 'sent_':
                sent_id = int(part.split('_')[1])
                new_input_token = enc(input_text_list[sent_id])
                if 'tumblr' in task_name :
                    if use_demo==True:
                        if len(new_input_token) >=45: ###tumblr数据有的太长，截断
                            new_input_token = new_input_token[:45]
                    else:
                        if len(new_input_token) >=400: ###tumblr数据有的太长，截断
                            new_input_token = new_input_token[:400] 
               
                new_tokens += new_input_token
                
            elif part[:6] == '+sent_':
                # Add space
                sent_id = int(part.split('_')[1])
                new_input_token = enc(' ' + input_text_list[sent_id])
                # if N is not None or N > 0:
                #     new_input_token = + [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token

            elif part[:6] == 'sent-_':
                # Delete the last token
                sent_id = int(part.split('_')[1])
                new_input_token = enc(input_text_list[sent_id][:-1])

                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token

            elif part[:6] == 'sentl_':
                # Lower case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_input_token = enc(text)
                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token

            elif part[:7] == '+sentl_':
                # Lower case the first token and add space 
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_input_token = enc(' ' + text)
                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token
    
            elif part[:7] == 'sentl-_':
                # Lower case the first token and discard the last token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].lower() + text[1:]
                new_input_token = enc(text[:-1])
                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token
            
            elif part[:6] == 'sentu_':
                # Upper case the first token
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_input_token = enc(text)
                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token

            elif part[:7] == '+sentu_':
                # Upper case the first token and add space
                sent_id = int(part.split('_')[1])
                text = input_text_list[sent_id]
                text = text[:1].upper() + text[1:]
                new_input_token = enc(' ' + text)
                # if N is not None or N > 0:
                #     new_input_token = [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_input_token
            ###<image>_'id'
            elif part[:8] == '<image>_':
                ###read_image
                sent_id = int(part.split('_')[1])
                raw, image = _read_image(image_path_list[sent_id])
                new_image = image
                image_pixel_values_list.append(new_image)
            
                for i in range(N):
                    new_tokens  += [I]
            elif part[:8] == 'caption_':
                ##encode image_caption
                # Add space
                sent_id = int(part.split('_')[1])
                new_caption_token = enc(' ' + image_caption_list[sent_id])
                # if N is not None or N > 0:
                #     new_input_token = + [special_token_mapping['sep+']] + new_input_token
                new_tokens += new_caption_token

            else:
                # Just natural language prompt
                part = part.replace('_', ' ') 
                # handle special case when T5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            if (part[:4] == 'sent' or part[1:5] == 'sent') and part!=' sentiment of text :' and part!=' sentiment':
                # If this part is the sentence, limit the sentence length
                sent_id = int(part.split('_')[1])
                if sent_id == 0:
                    if first_sent_limit is not None:
                        new_tokens = new_tokens[:first_sent_limit]
                else:
                    if other_sent_limit is not None:
                        new_tokens = new_tokens[:other_sent_limit]
            
            input_ids += new_tokens

            attention_mask += [1 for i in range(len(new_tokens))]
            token_type_ids += [segment_id for i in range(len(new_tokens))]

            if segment_plus_1_flag:
                segment_id += 1
    else:
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        token_type_ids = [0]

        for i in range(N):
            input_ids += [I]
            attention_mask += [1]
            token_type_ids += [0]

        for sent_id, input_text in enumerate(input_text_list):
            if input_text is None:
                # Do not have text_b
                continue
            if pd.isna(input_text) or input_text is None:
                # Empty input
                input_text = ''
            input_tokens = enc(input_text) + [tokenizer.sep_token_id]

            raw, image = _read_image(image_path_list[sent_id])
            new_image = image
            # if N is not None or N > 0:
            #     input_tokens = [I for i in range(N)] + [special_token_mapping['sep+']] + input_tokens

            input_ids += input_tokens

            image_pixel_values_list.append(new_image)

            attention_mask += [1 for i in range(len(input_tokens))]
            token_type_ids += [sent_id for i in range(len(input_tokens))]

        if 'T5' in type(tokenizer).__name__: # T5 does not have CLS token
            input_ids = input_ids[1:]
            attention_mask = attention_mask[1:]
            token_type_ids = token_type_ids[1:]

    
    
    # Padding
    if first_sent_limit is not None and len(input_ids) > max_length:
        # If using sentence limit, the total length still exceeds the maximum limit, report a warning
        logger.warn("Input exceeds max_length limit: {}".format(tokenizer.decode(input_ids)))

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
    
    image_token_mask = torch.tensor(input_ids) == image_token_id() ###tensor类型
    image_token_mask = image_token_mask.numpy().tolist() ###tensor-->list 元素为bool
    image_token_mask = [int(i) for i in image_token_mask] ###元素转为int

    # Find mask token
    if prompt:
        mask_pos = [input_ids.index(tokenizer.mask_token_id)]
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < max_length

    if len(image_pixel_values_list)>0:
        image_pixel_values_list = torch.stack(image_pixel_values_list).squeeze()

    result = {'input_ids': input_ids, 
              'attention_mask': attention_mask, 
              'image_pixel_values_list':image_pixel_values_list, 
              'image_token_mask': image_token_mask}

    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    if prompt:
        result['mask_pos'] = mask_pos

    return result



class FewShotDataset_AddCaption(torch.utils.data.Dataset):
    """Few-shot dataset."""

    def __init__(self, args, tokenizer, cache_dir=None, mode="train", use_demo=False, num_image_tokens=2, num_prompt_tokens=2 ,add_image=False, image_transform=None):
        self.args = args
        self.task_name = args.task_name
        self.processor = processors_mapping[args.task_name]
        self.tokenizer = tokenizer
        self.mode = mode
        self.add_image = add_image
        self.num_image_tokens = num_image_tokens
        self.num_prompt_tokens = num_prompt_tokens
        self.image_transform = image_transform

        # If not using demonstrations, use use_demo=True
        self.use_demo = use_demo
        if self.use_demo:
            logger.info("Use demonstrations")
        assert mode in ["train", "dev", "test"]

        # Get label list and (for prompt) label word list
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        if args.prompt:
            assert args.mapping is not None
            self.label_to_word = eval(args.mapping)

            for key in self.label_to_word:
                # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
                if self.label_to_word[key][0] not in ['<', '[', '.', ',']:
                    # Make sure space+word is in the vocabulary
                    assert len(tokenizer.tokenize(' ' + self.label_to_word[key])) == 1
                    self.label_to_word[key] = tokenizer._convert_token_to_id(tokenizer.tokenize(' ' + self.label_to_word[key])[0])
                else:
                    self.label_to_word[key] = tokenizer._convert_token_to_id(self.label_to_word[key])
                logger.info("Label {} to word {} ({})".format(key, tokenizer._convert_id_to_token(self.label_to_word[key]), self.label_to_word[key]))
            
            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                # Regression task
                # '0' represents low polarity and '1' represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ['0', '1']]
        else:
            self.label_to_word = None
            self.label_word_list = None

        # Multiple sampling: when using demonstrations, we sample different combinations of demonstrations during 
        # inference and aggregate the results by averaging the logits. The number of different samples is num_sample.
        if (mode == "train") or not self.use_demo:
            # We do not do multiple sampling when not using demonstrations or when it's the training mode 
            self.num_sample = 1
        else:
            self.num_sample = args.num_sample

        # If we use multiple templates, we also need to do multiple sampling during inference.
        if args.prompt and args.template_list is not None:
            logger.info("There are %d templates. Multiply num_sample by %d" % (len(args.template_list), len(args.template_list)))
            self.num_sample *= len(args.template_list)
                
        logger.info("Total num_sample for mode %s: %d" % (mode, self.num_sample))

        # Load cache
        # Cache name distinguishes mode, task name, tokenizer, and length. So if you change anything beyond these elements, make sure to clear your cache.
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )

        logger.info(f"Creating/loading examples from dataset file at {args.data_dir}")

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.support_examples, self.query_examples = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                # The support examples are sourced from the training set.
                self.support_examples = self.processor.get_train_examples(args.data_dir)

                if mode == "dev":
                    self.query_examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == "test":
                    self.query_examples = self.processor.get_test_examples(args.data_dir)
                else:
                    self.query_examples = self.support_examples

                start = time.time()
                torch.save([self.support_examples, self.query_examples], cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )
       
        # For filtering in using demonstrations, load pre-calculated embeddings
        if self.use_demo and args.demo_filter:
            split_name = ''
            if mode == 'train':
                split_name = 'train'
            elif mode == 'dev':
                if args.task_name == 'mnli':
                    split_name = 'dev_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'dev_mismatched'
                else:
                    split_name = 'val'
            elif mode == 'test':
                if args.task_name == 'mnli':
                    split_name = 'test_matched'
                elif args.task_name == 'mnli-mm':
                    split_name = 'test_mismatched'
                else:
                    split_name = 'test'
            else:
                raise NotImplementedError

            self.support_emb = np.load(os.path.join(args.data_dir, "train_{}.npy".format(args.demo_filter_model)))
            self.query_emb = np.load(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model)))
            logger.info("Load embeddings (for demonstration filtering) from {}".format(os.path.join(args.data_dir, "{}_{}.npy".format(split_name, args.demo_filter_model))))

            assert len(self.support_emb) == len(self.support_examples)
            assert len(self.query_emb) == len(self.query_examples)
 
        # Size is expanded by num_sample
        self.size = len(self.query_examples) * self.num_sample
        
        # Prepare examples (especially for using demonstrations)
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                # If training, exclude the current example. Else keep all.
                if self.use_demo and args.demo_filter:
                    # Demonstration filtering
                    candidate = [support_idx for support_idx in support_indices
                                   if support_idx != query_idx or mode != "train"]
                    sim_score = []
                    for support_idx in candidate:
                        sim_score.append((support_idx, util.pytorch_cos_sim(self.support_emb[support_idx], self.query_emb[query_idx])))
                    sim_score.sort(key=lambda x: x[1], reverse=True)
                    if self.num_labels == 1:
                        # Regression task
                        limit_each_label = int(len(sim_score) // 2 * args.demo_filter_rate)
                        count_each_label = {'0': 0, '1': 0}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] < limit_each_label:
                                count_each_label['0' if float(self.support_examples[support_idx].label) <= median_mapping[args.task_name] else '1'] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                    else:
                        limit_each_label = int(len(sim_score) // self.num_labels * args.demo_filter_rate)
                        count_each_label = {label: 0 for label in self.label_list}
                        context_indices = []

                        if args.debug_mode:
                            print("Query %s: %s" % (self.query_examples[query_idx].label, self.query_examples[query_idx].text_a)) # debug
                        for support_idx, score in sim_score:
                            if count_each_label[self.support_examples[support_idx].label] < limit_each_label:
                                count_each_label[self.support_examples[support_idx].label] += 1
                                context_indices.append(support_idx)
                                if args.debug_mode:
                                    print("    %.4f %s | %s" % (score, self.support_examples[support_idx].label, self.support_examples[support_idx].text_a)) # debug
                else:
                    # Using demonstrations without filtering
                    context_indices = [support_idx for support_idx in support_indices
                               if support_idx != query_idx or mode != "train"]

                # We'll subsample context_indices further later.
                self.example_idx.append((query_idx, context_indices, sample_idx))
      
        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode != "train":
            self.features = []
            _ = 0
            for query_idx, context_indices, bootstrap_idx in self.example_idx:
                # The input (query) example
                example = self.query_examples[query_idx]
                # The demonstrations
                # print('-------------------context_indices---------------------------')
                # print(context_indices)
                supports = self.select_context([self.support_examples[i] for i in context_indices], context_indices)
                supports2 = self.select_context([self.support_examples[i] for i in context_indices], context_indices)
                
                template = args.template
                # print('222222222222222222222222222222222')
                # print(template)

                if args.task_name == "mul_mvsa_single_contrastive_add_caption" or args.task_name =="mul_mvsa_multiple_contrastive_add_caption"  or args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or self.args.task_name =="mul_tumblr_contrastive_fusion_add_caption":
                    self.features.append(self.new_convert_fn(
                    example=example,
                    supports=supports,
                    supports2=supports2,
                    use_demo=self.use_demo,
                    label_list=self.label_list,
                    prompt=self.args.prompt,
                    template=template,
                    label_word_list=self.label_word_list,
                    verbose=True if _ == 0 else False,
                ))
                else:
                    self.features.append(self.convert_fn(
                        example=example,
                        supports=supports,
                        use_demo=self.use_demo,
                        label_list=self.label_list,
                        prompt=args.prompt,
                        template=template,
                        label_word_list=self.label_word_list,
                        verbose=True if _ == 0 else False,
                    ))

                

                _ += 1
        else:
            self.features = None

    def select_context(self, context_examples, context_indices):
        """
        Select demonstrations from provided examples.
        """
        max_demo_per_label = 1 ###每个标签只取一个demo
        counts = {k: 0 for k in self.label_list}
        if len(self.label_list) == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []

        if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
            # For GPT-3's in-context learning, we sample gpt3_in_context_num demonstrations randomly. 
            if self.use_demo and self.args.demo_filter:
                order =[]
                for i in range(len(context_examples)):
                    order.append(i)
            else:
                order = np.random.permutation(len(context_examples))
            for i in range(min(self.args.gpt3_in_context_num, len(order))):
                selection.append(context_examples[order[i]])
        else:
            # Our sampling strategy
            if self.use_demo and self.args.demo_filter:
                order = []
                for i in range(len(context_examples)):
                    order.append(i)
            else:
                order = np.random.permutation(len(context_examples))
            # print('-=======================order==============================')
            # print(order)
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~context_indices[order[0]]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(context_indices[order[0]])

            for i in order:
                label = context_examples[i].label
                if len(self.label_list) == 1:
                    # Regression
                    label = '0' if float(label) <= median_mapping[self.args.task_name] else '1'
                if counts[label] < max_demo_per_label:
                    selection.append(context_examples[i])
                    counts[label] += 1
                if sum(counts.values()) == len(counts) * max_demo_per_label:
                    break
        
            assert len(selection) > 0
        
        return selection

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, bootstrap_idx = self.example_idx[i]
            # The input (query) example
            example = self.query_examples[query_idx]
            # The demonstrations
            supports = self.select_context([self.support_examples[i] for i in context_indices], context_indices)
            supports2 = self.select_context([self.support_examples[i] for i in context_indices], context_indices)
            # print("+++++++++++++++++++++++supports+++++++++++++++++++++++")
            # print(supports)

            
            template = self.args.template
            # print('33333333333333333333333333333333333')
            # print(template)
            features = self.new_convert_fn(
                example=example,
                supports=supports,
                supports2=supports2,
                use_demo=self.use_demo,
                label_list=self.label_list,
                prompt=self.args.prompt,
                template=template,
                label_word_list=self.label_word_list,
                verbose=False,
            )
        else:
            features = self.features[i]
            
        return features

    def get_labels(self):
        return self.label_list


    def convert_fn(
        self,
        example,
        supports,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length    

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        ##label_list = ["negative", "neutral", "positive"]; label_map: {'negative': 0, 'neutral': 1, 'positive': 2}
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        if not use_demo:
            print('Do not use demo!!!!!!!!!!!!!!!!!')
            # No using demonstrations
            if not self.add_image:
                inputs = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                )
            else:
                
                inputs = tokenize_multipart_input_add_image(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=[example.image_path],
                    image_caption_list=[example.image_caption],
                    image_transform=self.image_transform,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    use_demo=self.use_demo,
                )
            features = OurInputFeatures(**inputs, label=example_label)

        else:
            # Using demonstrations

            # Max length
            if self.args.double_demo:
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
                max_length = 512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples, 
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's 
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example) # Input sentence list for query
            ###for image_path
            augmented_image_path = []
            query_image_path = [example.image_path]

            ###for image_caption
            augmented_image_caption = []
            query_image_caption = [example.image_caption]

            support_by_label = [[] for i in range(len(label_map))]
            support_image_path_by_label =  [[] for i in range(len(label_map))]
            support_image_caption_by_label =  [[] for i in range(len(label_map))]

            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                augmented_image_path = query_image_path
                augmented_image_caption = query_image_caption
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)

                    augmented_image_path += [support_example.image_path]
                    augmented_image_caption +=[support_example.image_caption]

                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' # Regression
                    support_labels.append(label_map[current_label])
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        # Regression
                        for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                            support_image_path_by_label[label_id] += [support_example.image_path]
                            support_image_caption_by_label[label_id] += [support_example.image_caption]

                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                            support_image_path_by_label[label_id] += [support_example.image_path]
                            support_image_caption_by_label[label_id] += [support_example.image_caption]

                augmented_example = query_text
                augmented_image_path = query_image_path
                augmented_image_caption = query_image_caption
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]
                    augmented_image_path += support_image_path_by_label[label_id]
                    augmented_image_caption += support_image_caption_by_label[label_id]
            # print("+++++++++++++++++++++++++++++augmented_example+++++++++++++++++++++++++++++++++++++++")
            # print(augmented_example)
            # Tokenization (based on the template)
            if not self.add_image:
                inputs = tokenize_multipart_input(
                    input_text_list=augmented_example,
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
                )
            else:
                # print('111111111111111111111111111111111111111111111111')
                # print('the template is {}'.format(template))
                '''
                ****************************template_list*********************************
                the template_list is ['', 'cls', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_0', '"_has', 'mask', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_1', '"_has', 'label_0', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_2', '"_has', 'label_1', '_emotion', 'sep+', '<image>', '<image>', 'sep+', '_The_sentense_"', 'sent_3', '"_has', 'label_2', '_emotion', 'sep+', '']
                '''
                inputs = tokenize_multipart_input_add_image(
                    input_text_list=augmented_example,
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=augmented_image_path,
                    image_caption_list=augmented_image_caption,
                    image_transform=self.image_transform,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels,
                    use_demo=self.use_demo,
                )
            features = OurInputFeatures(**inputs, label=example_label)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("the shape of image_pixel_values_list is {}".format(features.image_pixel_values_list.shape))
            logger.info("the shape of input_ids is %s" % np.shape(features.input_ids))
            logger.info("text: %s" % self.tokenizer.decode(features.input_ids))
        
        # print(features)
        # print("text: %s" % self.tokenizer.decode(features.input_ids))

        return features

    ###add data view for contrastive loss
    def new_convert_fn(
        self,
        example,
        supports,
        supports2,
        use_demo=False,
        label_list=None,
        prompt=False,
        template=None,
        label_word_list=None,
        verbose=False
    ):
        """
        Returns a list of processed "InputFeatures".
        """
        max_length = self.args.max_seq_length

        # Prepare labels
        label_map = {label: i for i, label in enumerate(label_list)} # Mapping the label names to label ids
        if len(label_list) == 1:
            # Regression
            label_map = {'0': 0, '1': 1}

        # Get example's label id (for training/inference)
        if example.label is None:
            example_label = None
        elif len(label_list) == 1:
            # Regerssion
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        # Prepare other features
        if not use_demo:
            #####  the 2nd view of input for contrastive learning.
            #####  the 2nd view of input is by replacing with a different template2
            template2 = self.args.template2
            # No using demonstrations
            if not self.add_image:
                inputs = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                )

               
                view2_inputs = tokenize_multipart_input(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template2,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                )
            
            else:
                inputs = tokenize_multipart_input_add_image(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=[example.image_path],
                    image_caption_list=[example.image_caption],
                    image_transform=self.image_transform,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    use_demo=self.use_demo,
                )


                view2_inputs = tokenize_multipart_input_add_image(
                    input_text_list=input_example_to_tuple(example),
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=[example.image_path],
                    image_caption_list=[example.image_caption],
                    image_transform=self.image_transform,
                    template=template2,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    use_demo=self.use_demo,
                )
           

           
            if self.args.task_name == "mul_mvsa_single_contrastive_add_caption" or self.args.task_name =="mul_mvsa_multiple_contrastive_add_caption"  or self.args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or self.args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or self.args.task_name =="mul_tumblr_contrastive_fusion_add_caption":
                view12_input_ids = np.stack((inputs['input_ids'], view2_inputs['input_ids']), axis=0) ##[2, 256]
                view12_attention_mask = np.stack((inputs['attention_mask'], view2_inputs['attention_mask']), axis=0)
                view12_mask_pos = np.stack((inputs['mask_pos'], view2_inputs['mask_pos']), axis=0)
                view12_image_pixel_values_list = np.stack((inputs['image_pixel_values_list'], view2_inputs['image_pixel_values_list']), axis=0) 
                view12_image_pixel_values_list = np.expand_dims(view12_image_pixel_values_list, axis=1) ##[2, 3, 256, 256]-->[2, 1, 3, 256, 256]
                view12_image_token_mask = np.stack((inputs['image_token_mask'], view2_inputs['image_token_mask']), axis=0)

                features = OurInputFeatures(input_ids=view12_input_ids, 
                                                attention_mask=view12_attention_mask, 
                                                token_type_ids= None,
                                                label=example_label,
                                                mask_pos=view12_mask_pos,
                                                label_word_list=None,
                                                image_pixel_values_list=view12_image_pixel_values_list, 
                                                raw_image=None,
                                                image_token_mask=view12_image_token_mask)
            else:  
                features = OurInputFeatures(inputs['input_ids'], inputs['attention_mask'], None, example_label, inputs['mask_pos'],None, 
                                            inputs['image_pixel_values_list'], None, inputs['image_token_mask'],
                                            inputs['input_ids'], inputs['attention_mask'], None, example_label, inputs['mask_pos'], None, 
                                            inputs['image_pixel_values_list'], None, inputs['image_token_mask'],
                                            view2_inputs['input_ids'], view2_inputs['attention_mask'], None, example_label, view2_inputs['mask_pos'], None, 
                                            view2_inputs['image_pixel_values_list'], None, view2_inputs['image_token_mask'],)




        else:
           # Using demonstrations

            # Max length
            if self.args.double_demo:
                # When using demonstrations, double the maximum length
                # Note that in this case, args.max_seq_length is the maximum length for a single sentence
                max_length = max_length * 2
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                # When using GPT-3's in-context learning, take the maximum tokenization length of the model (512)
                max_length = 512

            # All input sentences, including the query and the demonstrations, are put into augmented_examples, 
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's 
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example = []
            query_text = input_example_to_tuple(example) # Input sentence list for query
            ###for image_path
            augmented_image_path = []
            query_image_path = [example.image_path]

            ###for image_caption
            augmented_image_caption = []
            query_image_caption = [example.image_caption]

            support_by_label = [[] for i in range(len(label_map))]
            support_image_path_by_label =  [[] for i in range(len(label_map))]
            support_image_caption_by_label =  [[] for i in range(len(label_map))]

            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                support_labels = []
                augmented_example = query_text
                augmented_image_path = query_image_path
                for support_example in supports:
                    augmented_example += input_example_to_tuple(support_example)
                    augmented_image_path += [support_example.image_path]
                    augmented_image_caption += [support_example.image_caption]

                    current_label = support_example.label
                    if len(label_list) == 1:
                        current_label = '0' if float(current_label) <= median_mapping[self.args.task_name] else '1' # Regression
                    support_labels.append(label_map[current_label])
   
            else:
               # Group support examples by label
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        # Regression
                        for support_example in filter(lambda s: ('0' if float(s.label) <= median_mapping[self.args.task_name] else '1') == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                            support_image_path_by_label[label_id] += [support_example.image_path]
                            support_image_caption_by_label[label_id] += [support_example.image_caption]

                    else:
                        for support_example in filter(lambda s: s.label == label_name, supports):
                            support_by_label[label_id] += input_example_to_tuple(support_example)
                            support_image_path_by_label[label_id] += [support_example.image_path]
                            support_image_caption_by_label[label_id] += [support_example.image_caption]

                augmented_example = query_text
                augmented_image_path = query_image_path
                augmented_image_caption = query_image_caption
                for label_id in range(len(label_map)):
                    augmented_example += support_by_label[label_id]
                    augmented_image_path += support_image_path_by_label[label_id]
                    augmented_image_caption += support_image_caption_by_label[label_id]
            # print("+++++++++++++++++++++++++++++augmented_image_path+++++++++++++++++++++++++++++++++++++++")
            # print(augmented_image_path)

            if not self.add_image:
                inputs = tokenize_multipart_input(
                    input_text_list=augmented_example,
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
                )
            else:

                inputs = tokenize_multipart_input_add_image(
                    input_text_list=augmented_example,
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=augmented_image_path,
                    image_caption_list=augmented_image_caption,
                    image_transform=self.image_transform,
                    template=template,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels,
                    use_demo=self.use_demo,
                )


            ''' +++++++++++++++++++++++++ the 2nd view of input for contrastive learning.++++++++++++++++++++++++++++++++++++'''
            #####  the 2nd view of input for contrastive learning.
            #####  the 2nd view of input is by replacing with a different template2
            template2 = self.args.template2
            # print("+++++++++++++++++++++++++++++template2 before +++++++++++++++++++++++++++++++++++++++")
            # print(template2)
           
            # print("+++++++++++++++++++++++++++++template2 after +++++++++++++++++++++++++++++++++++++++")
            # print(template2)

            ###### augmented_example for view2

            # All input sentences, including the query and the demonstrations, are put into augmented_examples,
            # and are numbered based on the order (starting from 0). For single sentence tasks, the input (query)
            # is the sentence 0; for sentence-pair tasks, the input (query) is the sentence 0 and 1. Note that for GPT-3's
            # in-context learning, the input (query) might be at the end instead of the beginning (gpt3_in_context_head)
            augmented_example2 = []
            query_text2 = input_example_to_tuple(example) # Input sentence list for query

             ###for image_path
            augmented_image_path2 = []
            query_image_path2 = [example.image_path]

            ###for image_caption
            augmented_image_caption2 = []
            query_image_caption2 = [example.image_caption]

            support_by_label2 = [[] for i in range(len(label_map))]
            support_image_path_by_label2 =  [[] for i in range(len(label_map))]
            support_image_caption_by_label2 =  [[] for i in range(len(label_map))]


                
            if self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail:
                pass
            else:
                # Group support examples by label
                for label_name, label_id in label_map.items():
                    if len(label_list) == 1:
                        # Regression
                        pass
                    else:
                        for support_example2 in filter(lambda s: s.label == label_name, supports2):
                            support_by_label2[label_id] += input_example_to_tuple(support_example2)
                            support_image_path_by_label2[label_id] += [support_example2.image_path]
                            support_image_caption_by_label2[label_id] += [support_example2.image_caption]

                augmented_example2 = query_text2
                augmented_image_path2 = query_image_path2
                augmented_image_caption2 = query_image_caption2
                for label_id in range(len(label_map)):
                    augmented_example2 += support_by_label2[label_id]
                    augmented_image_path2 += support_image_path_by_label2[label_id]
                    augmented_image_caption2 += support_image_caption_by_label2[label_id]
                # print('-------------------augmented_image_path2--------------------------')
                # print(augmented_image_path2)
                # print(augmented_example2)

            # Tokenization (based on the template)
            if not self.add_image:

                view2_inputs = tokenize_multipart_input(
                    input_text_list=augmented_example2,     ##### for ablation study
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    template=template2,     ### use the sampled template  ##### for ablation study
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels
                )
            
            else:

                view2_inputs = tokenize_multipart_input_add_image(
                    input_text_list=augmented_example2,
                    max_length=max_length,
                    tokenizer=self.tokenizer,
                    task_name=self.args.task_name,
                    prompt=prompt,
                    add_image=self.add_image,
                    num_image_tokens=self.num_image_tokens,
                    num_prompt_tokens=self.num_prompt_tokens,
                    image_path_list=augmented_image_path2,
                    image_caption_list=augmented_image_caption2,
                    image_transform=self.image_transform,
                    template=template2,
                    label_word_list=label_word_list,
                    first_sent_limit=self.args.first_sent_limit,
                    other_sent_limit=self.args.other_sent_limit,
                    truncate_head=self.args.truncate_head,
                    gpt3=self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail,
                    support_labels=None if not (self.args.gpt3_in_context_head or self.args.gpt3_in_context_tail) else support_labels,
                    use_demo=self.use_demo,
                )



            
            # else:
            if self.args.task_name == "mul_mvsa_single_contrastive_add_caption" or self.args.task_name =="mul_mvsa_multiple_contrastive_add_caption"  or self.args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or self.args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or self.args.task_name =="mul_tumblr_contrastive_fusion_add_caption":
                view12_input_ids = np.stack((inputs['input_ids'], view2_inputs['input_ids']), axis=0) ##[2, 256]
                view12_attention_mask = np.stack((inputs['attention_mask'], view2_inputs['attention_mask']), axis=0)
                view12_mask_pos = np.stack((inputs['mask_pos'], view2_inputs['mask_pos']), axis=0)
                view12_image_pixel_values_list = np.stack((inputs['image_pixel_values_list'], view2_inputs['image_pixel_values_list']), axis=0) ##[2, 4, 3, 256, 256]
                view12_image_token_mask = np.stack((inputs['image_token_mask'], view2_inputs['image_token_mask']), axis=0)

                features = OurInputFeatures(input_ids=view12_input_ids, 
                                                attention_mask=view12_attention_mask, 
                                                token_type_ids= None,
                                                label=example_label,
                                                mask_pos=view12_mask_pos,
                                                label_word_list=None,
                                                image_pixel_values_list=view12_image_pixel_values_list, 
                                                raw_image=None,
                                                image_token_mask=view12_image_token_mask)
            else:
                features = OurInputFeatures(inputs['input_ids'], inputs['attention_mask'], None, example_label, inputs['mask_pos'],None, 
                                            inputs['image_pixel_values_list'], None, inputs['image_token_mask'],
                                            inputs['input_ids'], inputs['attention_mask'], None, example_label, inputs['mask_pos'], None, 
                                            inputs['image_pixel_values_list'], None, inputs['image_token_mask'],
                                            view2_inputs['input_ids'], view2_inputs['attention_mask'], None, example_label, view2_inputs['mask_pos'], None, 
                                            view2_inputs['image_pixel_values_list'], None, view2_inputs['image_token_mask'],)

        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("features: %s" % features)
            logger.info("the shape of image_pixel_values_list is {}".format(features.image_pixel_values_list.shape))
            logger.info("the shape of input_ids is {}".format(np.shape(features.input_ids)))
            for i in range(len(features.input_ids)):
                logger.info("text: %s" % self.tokenizer.decode(features.input_ids[i]))
            # logger.info("view2_text: %s" % self.tokenizer.decode(features.view2_input_ids))
        
        # print('+++++++++++++++the guid is {}+++++++++++++++++++++++++++'.format(example.guid))
        # print(features)
        # print("text: %s" % self.tokenizer.decode(features.input_ids))
        # print("view2_text: %s" % self.tokenizer.decode(features.view2_input_ids))

        return features


from transformers import HfArgumentParser, TrainingArguments, set_seed
from dataclasses import dataclass, field
import sys
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import RobertaTokenizer
@dataclass
class ModelArguments:
    """
     Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
     )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )

    add_image: bool = field(
        default=False,
        metadata={"help": "Whether to add image in the load dataset)."}
    )
    image_model_name: str = field(
        default='microsoft/resnet-50',
        metadata={"help": "the name of image_model"}
    )

    num_image_tokens: Optional[int] = field(
        default=2,
        metadata={"help": "Number of the image tokens in the tezxt_image inputs"}
    )

    frozen_text_encoder:bool = field(
        default=False,
        metadata={"help": "whether to frozen the text encoder."}
    )

    frozen_image_encoder:bool = field(
        default=False,
        metadata={"help": "whether to frozen the image encoder."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
    
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"}
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"}
    )

    gpt3_in_context_num: int = field(
        default=32,
        metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: list = field(
        default=None,
        metadata={"help": "(DO NOT List of templates (only initialized after the program starts."}
    )
    

    template_list_location: str = field(
        default='/home/xiaocui/code/LM-CoCop/LM-BFF/my_auto_template/MVSA_Single/32-87.sort.txt',
        metadata={"help": "the template list file location"}
    )

    template_list_number: int = field(
        default=20,
        metadata={"help": "the length of template list"}
    )

    template_list_new: list = field(
        default=None,
        metadata={"help": "the template list"}
    )


    template2: str = field(
        default=None,
        metadata={"help": "Template2 for contrastive learning"}
    )



@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )
    supcon_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "the learning rate for SupCon Loss"}
    )




class Collate():
    def __init__(self):
       
        self.image_mask_num = 0
       

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        input_ids = [torch.LongTensor(b['input_ids']) for b in batch_data]
        attention_mask = [torch.LongTensor([b['attention_mask'] for b in batch_data])]
        mask_pos = torch.LongTensor([b['mask_pos'] for b in batch_data])
        # text_no_prompt_ids = [torch.LongTensor(b[3]) for b in batch_data]
        # text_no_prompt_attention_mask = [torch.LongTensor([b[4] for b in batch_data])]
        label = torch.LongTensor([b['label'] for b in batch_data])


        data_length = [text.size(0) for text in input_ids]



        return input_ids, attention_mask,  mask_pos, label
    
def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #### print the training args
    for arg in vars(training_args):
        print(arg, getattr(training_args, arg))

    # template list
    if data_args.template_list_new == None:
        assert data_args.template_list_location is not None
        data_args.template_list_new = []
        count = 0
        with open(data_args.template_list_location) as fp:
            Lines = fp.readlines()
            if data_args.template_path is not None: ###如果使用排名第一的sort后的templete, 去掉用掉的第一条template
                Lines = Lines[1:]
            else:
                Lines = Lines
            for line in Lines:
                count += 1
                if count > data_args.template_list_number:
                    break
                line = line.replace('*cls*', '*cls*<image>_0*is*caption_0*sep+*')
                ##'*cls*<image>_0*_is_*caption_0*sep+**sent_0*_It_was*mask*.*sep+*'

                data_args.template_list_new.append(line.strip())
    else:
        data_args.template_list_new = ["*cls**sent_0*_The_movie_is*mask*.*sep+*",
                                       "*cls**sent_0*_The_music_is*mask*.*sep+*",
                                       "*cls**sent_0*_But_it_is*mask*.*sep+*",
                                       "*cls*_This_one_is*mask*.*+sent_0**sep+*",
                                       "*cls**sent_0*_Is_it*mask*?*sep+*",
                                       "*cls**sent_0*_A*mask*_movie.*sep+*",
                                       "*cls**sent_0*_Its*mask*.*sep+*",
                                       "*cls**sent_0*_It_is_just*mask*.*sep+*",
                                       "*cls**sent_0*_All_in_all*mask*.*sep+*",
                                       "*cls**sent_0*_The_ending_is*mask*.*sep+*",
                                       "*cls**sent_0*_It_really_is*mask*.*sep+*",
                                       "*cls**sent_0*_That's*mask*!*sep+*",
                                       "*cls**sent_0*_It’s*mask*!*sep+*",
                                       "*cls**sent_0*_I'm*mask*.*sep+*",
                                       "*cls**sent_0*_In_other_words*mask*.*sep+*",
                                       "*cls**sent_0*_Its_just*mask*.*sep+*",
                                       "*cls**sent_0*_A*mask*_watch.*sep+*",
                                       "*cls**sent_0*_It's*mask*.*sep+*",
                                       "*cls**sent_0*_It's*mask*!*sep+*",
                                       "*cls**sent_0*_The_film_is*mask*.*sep+*"]
    
    

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(data_args.template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    if model_args.add_image == True:
                        data_args.template = data_args.template.replace('*cls*', '*cls*<image>_0*sep+*')

                    data_args.template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        output_mode = output_modes_mapping[data_args.task_name]
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail: 
            logger.info("Automatically convert the template to GPT-3's in-context learning.")
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ''
            old_template = old_template.replace('*cls*', '')
            print('----------------------old_template----------------------')
            print(old_template)
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ''
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * instance_id + sent_id))
                    # Replace <image> 
                    sub_template = sub_template.replace("<image>_{}".format(sent_id), "<image>_{}".format(sent_num + sent_num * instance_id + sent_id))
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*labelx_{}*".format(instance_id))
               
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template # Put context at the end
                else:
                    new_template = sub_template + new_template # Put context at the beginning
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ''
                    old_template = old_template.replace('*cls*', '')
                    
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ''
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                            # Replace <image> 
                            sub_template = sub_template.replace("<image>_{}".format(sent_id), "<image>_{}".format(sent_num + sent_num * label_id + sent_id))
                        # Replace mask
                        sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                       

                        new_template = new_template + sub_template
                    logger.info("| {} => {}".format(data_args.template_list[i], new_template))
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                        # Replace <image> 
                        sub_template = sub_template.replace("<image>_{}".format(sent_id), "<image>_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))

                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(data_args.template, new_template))
                data_args.template = new_template

                #deal with template 2
                #####  Automatically convert the template to using demonstrations.
                if data_args.template_list_new is not None:
                    template_list_new_temp = []
                    for i in range(len(data_args.template_list_new)):
                        old_template = data_args.template_list_new[i]
                        new_template = old_template + ''
                        old_template = old_template.replace('*cls*', '')
                        # Single sentence or sentence pair?
                        sent_num = 1
                        if "_1" in old_template:
                            sent_num = 2
                        for label_id in range(num_labels):
                            sub_template = old_template + ''
                            # Replace sent id
                            for sent_id in range(sent_num):
                                sub_template = sub_template.replace("_{}".format(sent_id),
                                                                    "_{}".format(sent_num + sent_num * label_id + sent_id))
                                # Replace <image> 
                                sub_template = sub_template.replace("<image>_{}".format(sent_id), "<image>_{}".format(sent_num + sent_num * label_id + sent_id))
                            # Replace mask
                            sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                            new_template = new_template + sub_template
                        logger.info("| {} => {}".format(data_args.template_list_new[i], new_template))
                        template_list_new_temp.append(new_template)
                    data_args.template_list_new = template_list_new_temp

    data_args.template2 = random.choice(data_args.template_list_new) 
    print('++++++++++++++++++data_args.template2 is {}++++++++++++++++++++++++++++++++++++++'.format(data_args.template2))   
    
    
    image_transform = get_image_transform('nf_resnet50')

    # Create tokenizer
    special_tokens = []
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
            )

    train_dataset = (
        FewShotDataset_AddCaption(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type), add_image=model_args.add_image,
        image_transform=image_transform)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size= 16, shuffle=False, collate_fn=Collate())

    '''
        input_ids: List[int]
        attention_mask: Optional[List[int]] = None
        token_type_ids: Optional[List[int]] = None
        label: Optional[Union[int, float]] = None
        mask_pos: Optional[List[int]] = None # Position of the mask token
        label_word_list: Optional[List[int]] = None # Label word mapping (dynamic)
    '''
    for i, features in enumerate(train_loader):
       print(features)


if __name__ == "__main__":
    main()