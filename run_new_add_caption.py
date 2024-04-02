"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch

import numpy as np
from typing import List
import transformers
from transformers import RobertaTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.dataset_new import FewShotDataset_new
from src.dataset import FewShotDataset
from src.dataset_new_add_view_and_caption_for_me import FewShotDataset_AddCaption
from src.dataset_new_add_view_and_caption_for_mabsa_me import FewShotDataset_AddCaption as FewShotDataset_AddCaption_MABSA

from src.models import BertForPromptFinetuning, RobertaForPromptFinetuning, resize_token_type_embeddings
from src.multimodal_models import ResNetRobertaForPromptFinetuning, add_image_token, ContrastiveResNetRobertaForPromptFinetuning, FusionContrastiveResNetRobertaForPromptFinetuning, FusionContrastiveResNetBertForPromptFinetuning

from src.trainer import Trainer

from src.trainer_add_views_fusion import Trainer as Trainer_Contrastive_Fusion

from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageFile, UnidentifiedImageError
from timm.data.transforms_factory import create_transform
from torchvision.transforms import Compose, Lambda
from transformers import GPT2Tokenizer, AutoFeatureExtractor, CLIPFeatureExtractor

import random
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
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

    num_prompt_tokens: Optional[int] = field(
        default=2,
        metadata={"help": "Number of the prompt tokens in the tezxt_image inputs"}
    )


    frozen_text_encoder:bool = field(
        default=False,
        metadata={"help": "whether to frozen the text encoder."}
    )

    frozen_image_encoder:bool = field(
        default=False,
        metadata={"help": "whether to frozen the image encoder."}
    )
    use_bayes_fusion:bool = field(
        default=True,
        metadata={"help": "whether to use bayes_fusion."}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    percent: Optional[int] = field(
        default=1,
        metadata={"help": "Percent of training instances"}
    )
    log_dir: str = field(
        default='log_new',
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=1,
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
    template_list: List[str] = field(
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

    template_list_new: List[str] = field(
        default=None,
        metadata={"help": "the template list"}
    )

    template2: str = field(
        default=None,
        metadata={"help": "Template2 for contrastive learning"}
    )
    # max_seq_length: int = field(
    #     default=256,
    #     metadata={"help": "the length of text"}
    # )



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
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "evaluate model during training"}
    )
    supcon_learning_rate: float = field(
        default=1e-5,
        metadata={"help": "the learning rate for SupCon Loss"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "the num_workers of dataloder"}
    )
    use_contrastive: bool = field(
        default=False,
        metadata={"help": "use contrastive module"}
    )
    # train_batch_size: int = field(
    #     default=8,
    #     metadata={"help": "train batch_size"}
    # )
    # per_device_eval_batch_size: int = field(
    #     default=32,
    #     metadata={"help": "eval batch_size"}
    # )




def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.local_rank = -1 
    #### print the training args
    for arg in vars(training_args):
        print(arg, getattr(training_args, arg))
    

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
    if model_args.few_shot_type == 'prompt-demo':
        logger.info("Automatically convert the template to using demonstrations.")
       
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
                # Replace caption_{id}
                if 'caption' in data_args.task_name:
                    sub_template = sub_template.replace("caption_{}".format(sent_id), "caption_{}".format(sent_num + sent_num * label_id + sent_id))
            # Replace mask
            sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))

            new_template = new_template + sub_template
        logger.info("| {} => {}".format(data_args.template, new_template))
        data_args.template = new_template

        #deal with template 2
        #####  Automatically convert the template to using demonstrations.
        if data_args.template2 != None:
            old_template2 = data_args.template2
            new_template2 = old_template2 + ''
            old_template2 = old_template2.replace('*cls*', '')
            
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template2:
                sent_num = 2
            for label_id in range(num_labels):
                sub_template2 = old_template2 + ''
                # Replace sent id
                for sent_id in range(sent_num):
                    sub_template2 = sub_template2.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace <image> 
                    sub_template2 = sub_template2.replace("<image>_{}".format(sent_id), "<image>_{}".format(sent_num + sent_num * label_id + sent_id))
                    # Replace caption_{id}
                    if 'caption' in data_args.task_name:
                        sub_template2 = sub_template2.replace("caption_{}".format(sent_id), "caption_{}".format(sent_num + sent_num * label_id + sent_id))
                # Replace mask
                sub_template2 = sub_template2.replace("*mask*", "*label_{}*".format(label_id))

                new_template2= new_template2 + sub_template2
            logger.info("| {} => {}".format(data_args.template2, new_template2))
            data_args.template2 = new_template2
            
        else:
            print('please check you template2!!!!!!!')

    print('++++++++++++++++++data_args.template is {}++++++++++++++++++++++++++++++++++++++'.format(data_args.template))   
    print('++++++++++++++++++data_args.template2 is {}++++++++++++++++++++++++++++++++++++++'.format(data_args.template2))   
    
    if not training_args.do_train:
        model_args.config_name = os.path.join(training_args.output_dir, 'config.json')
    
    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        
    )
    
    if data_args.task_name == "mul_mvsa_single" or  data_args.task_name =="mul_mvsa_multiple":
        if 'prompt' in model_args.few_shot_type:
            if config.model_type == 'roberta':
                # model_fn = MultimodalRobertaForPromptFinetuning
                model_fn =ResNetRobertaForPromptFinetuning
            elif config.model_type == 'bert':
                print('BERT with adding image do not been completed!!!!')
            else:
                raise NotImplementedError
        elif model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError
    
    
    elif data_args.task_name == "mul_mvsa_single_contrastive_fusion" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion" or data_args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or data_args.task_name =="mul_tumblr_contrastive_fusion_add_caption" or data_args.task_name == "mul_t2015_contrastive_fusion_add_caption" or data_args.task_name =="mul_t2017_contrastive_fusion_add_caption" or data_args.task_name =="mul_masad_contrastive_fusion_add_caption":
        if 'prompt' in model_args.few_shot_type:
            if config.model_type == 'roberta':
                model_fn =FusionContrastiveResNetRobertaForPromptFinetuning
            elif config.model_type == 'bert':
                model_fn = FusionContrastiveResNetBertForPromptFinetuning
            else:
                raise NotImplementedError
        elif model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError
    

    else:
        if 'prompt' in model_args.few_shot_type:
            if config.model_type == 'roberta':
                model_fn = RobertaForPromptFinetuning
            elif config.model_type == 'bert':
                model_fn = BertForPromptFinetuning
            else:
                raise NotImplementedError
        elif model_args.few_shot_type == 'finetune':
            model_fn = AutoModelForSequenceClassification
        else:
            raise NotImplementedError
    special_tokens = []

    image_transform = get_image_transform('nf_resnet50')
    

    # Create tokenizer
    if model_args.model_name_or_path == 'roberta-large' or model_args.tokenizer_name=='roberta-large' or model_args.model_name_or_path == 'roberta-base' or model_args.tokenizer_name=='roberta-base':
        tokenizer = RobertaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
            )
        
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            additional_special_tokens=special_tokens,
            cache_dir=model_args.cache_dir,
        )
        
    tokenizer.add_special_tokens(SPECIAL_TOKEN_DICT) 
    
    # import pdb; pdb.set_trace()
    # Get our special datasets.
    if data_args.task_name == "mul_mvsa_single" or data_args.task_name =="mul_mvsa_multiple":
        print('+++++++++++++++++loading the Mul_MVSA dataset++++++++++++++++=')
        train_dataset = (
            FewShotDataset_new(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
        )
        eval_dataset = (
            FewShotDataset_new(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_eval
            else None
        )
        test_dataset = (
            FewShotDataset_new(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_predict
            else None
        )
    
    elif data_args.task_name == "mul_mvsa_single_contrastive_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_add_caption"  or data_args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or data_args.task_name =="mul_tumblr_contrastive_fusion_add_caption": 
        print('+++++++++++++++++loading the Mul_MVSA_Contrastive_Add_Caption dataset++++++++++++++++=')
        print(model_args.few_shot_type)
        # if not training_args.do_train:
        
        train_dataset = (
            FewShotDataset_AddCaption(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
        )
        eval_dataset = (
            FewShotDataset_AddCaption(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_eval
            else None
        )
        test_dataset = (
            FewShotDataset_AddCaption(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_predict
            else None
        )
    elif data_args.task_name == "mul_t2015_contrastive_add_caption" or data_args.task_name =="mul_t2017_contrastive_add_caption" or  data_args.task_name =="mul_masad_contrastive_add_caption" or data_args.task_name == "mul_t2015_contrastive_fusion_add_caption" or data_args.task_name =="mul_t2017_contrastive_fusion_add_caption" or data_args.task_name =="mul_masad_contrastive_fusion_add_caption" : 
        train_dataset = (
            FewShotDataset_AddCaption_MABSA(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
        )
        eval_dataset = (
            FewShotDataset_AddCaption_MABSA(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_eval
            else None
        )
        test_dataset = (
            FewShotDataset_AddCaption_MABSA(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type),
            num_image_tokens=model_args.num_image_tokens,
            num_prompt_tokens=model_args.num_prompt_tokens,
            add_image=model_args.add_image,
            image_transform=image_transform)
            if training_args.do_predict
            else None
        )
    else:
        train_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
        )
        eval_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
            if training_args.do_eval
            else None
        )
        test_dataset = (
            FewShotDataset(data_args, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
            if training_args.do_predict
            else None
        )

    set_seed(training_args.seed)

    if data_args.task_name == "mul_mvsa_single" or data_args.task_name =="mul_mvsa_multiple":
        model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        opt=model_args,
        )
        # num_embeddings = model.get_input_embeddings().num_embeddings
        # model._resize_token_embeddings(new_num_tokens=(num_embeddings+1))
        # model.get_output_embeddings()

        # print('+++++++++++++++++++++++the new_num_embeddings is {}'.format(model.get_input_embeddings().num_embeddings))
    elif data_args.task_name == "mul_mvsa_single_contrastive" or data_args.task_name =="mul_mvsa_multiple_contrastive" or  data_args.task_name =="mul_tumblr_contrastive" or data_args.task_name == "mul_mvsa_single_contrastive_fusion" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion" or data_args.task_name == "mul_mvsa_single_contrastive_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_add_caption"  or data_args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or data_args.task_name =="mul_tumblr_contrastive_fusion_add_caption" or data_args.task_name == "mul_t2015_contrastive_add_caption" or data_args.task_name =="mul_t2017_contrastive_add_caption" or  data_args.task_name =="mul_masad_contrastive_add_caption" or data_args.task_name == "mul_t2015_contrastive_fusion_add_caption" or data_args.task_name =="mul_t2017_contrastive_fusion_add_caption" or data_args.task_name =="mul_masad_contrastive_fusion_add_caption" or data_args.task_name == "mul_t2015_contrastive" or data_args.task_name =="mul_t2017_contrastive" or  data_args.task_name =="mul_masad_contrastive":
        # import ipdb; ipdb.set_trace()
        if training_args.do_train:
            model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            opt=model_args,
            )
        else:
            checkpoint = torch.load(os.path.join(training_args.output_dir, 'pytorch_model.bin'))
            model = model_fn(config=config, opt=model_args)
            print(model)
            model.load_state_dict(checkpoint)
            model = model.to(training_args.device)
            
            model.eval()

    else:
        model = model_fn.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    # model.resize_token_embeddings(len(tokenizer)) 
   
    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == 'bert':
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            
            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn
    
   
    if data_args.task_name == "mul_mvsa_single_contrastive_fusion" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion" or data_args.task_name == "mul_mvsa_single_contrastive_fusion_add_caption" or data_args.task_name =="mul_mvsa_multiple_contrastive_fusion_add_caption" or data_args.task_name =="mul_tumblr_contrastive_fusion_add_caption" or data_args.task_name == "mul_t2015_contrastive_fusion_add_caption" or data_args.task_name =="mul_t2017_contrastive_fusion_add_caption" or data_args.task_name =="mul_masad_contrastive_fusion_add_caption":
        print('!!!!!!!!!!!!!!!!the trainer is Trainer_Contrastive_Fusion!!!!!!!!!!!!!!!!!')
        trainer = Trainer_Contrastive_Fusion(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )

    else:
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn(data_args.task_name)
        )

    # Training
    # import ipdb; ipdb.set_trace()
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(training_args.output_dir)
 
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
            torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
            torch.save(data_args, os.path.join(training_args.output_dir, "data_args.bin"))
        
        # Reload the best checkpoint (for eval)
      
        # import pdb; pdb.set_trace() ##debug
        model = model_fn.from_pretrained(training_args.output_dir, opt=model_args)

        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer
    
    
    
    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[eval_dataset.args.task_name + '_dev_' + key] = value
            eval_results.update(eval_result)

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                FewShotDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test",use_demo=('demo' in model_args.few_shot_type))
            )

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                    if training_args.save_logit:
                        predictions = output.predictions
                        num_logits = predictions.shape[-1]
                        logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)
                        np.save(os.path.join(training_args.save_logit_dir, "{}-{}-{}.npy".format(test_dataset.task_name, training_args.model_id, training_args.array_id)), logits)

            test_results.update(test_result)

    
    if not os.path.exists(data_args.log_dir):
        os.makedirs(data_args.log_dir)

    lock_name = os.path.join(data_args.log_dir, 'log.lock')
    
    log_name = os.path.join(data_args.log_dir, 'log')
    

    with FileLock(lock_name):
        with open(log_name, 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')
    
    return eval_results

if __name__ == "__main__":
    main()
