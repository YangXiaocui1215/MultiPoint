"""Custom models for few-shot learning specific operations."""

import pdb
import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead, RobertaPreTrainedModel, RobertaPreTrainedModel

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, AutoModel, CLIPVisionModel, CLIPVisionConfig


import timm

from transformers import BertConfig, BertForPreTraining, RobertaConfig
import copy
import torchvision.models as cv_models
import math
from src.pre_model import RobertaEncoder
from packaging import version
import logging
logger = logging.getLogger(__name__)

def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError



'''
增加图像模态来进行微调
'''

def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)

class TextModel(nn.Module):
    def __init__(self, opt, model_config):
        super(TextModel, self).__init__()

        if opt.text_model == 'bert-base':
            self.config = model_config
            self.model = BertModel(self.config)

        elif opt.text_model == 'roberta-large':
            self.config = model_config
            self.model = RobertaModel(self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output

class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        # self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        # self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        # self.output_dim = self.resnet_encoder[7][2].conv3.out_channels
        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # for param in self.resnet.parameters():
        #     if opt.fixed_image_model:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_cls = self.resnet_encoder(images)
        # image_encoder = self.conv_output(image_encoder)
        # image_cls = self.resnet_avgpool(image_encoder)
        # image_cls = torch.flatten(image_cls, 1)
        return image_cls

class MultimodalRobertaForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config, opt):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt, config)
        self.image_model = ImageModel(opt)
        self.lm_head = RobertaLMHead(config)

        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        # For auto label search.
        self.return_full_softmax = None

        self.label_word_list = None

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)

        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )

        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 3)
        )

    def forward(self, input_ids, attention_mask, 
                image_inputs=None, image_text_attention_mask=None,
                mask_pos=None, labels=None):


        text_encoder = self.text_model(input_ids, attention_mask=attention_mask)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)
        image_encoder, image_cls = self.image_model(image_inputs)
        if self.image_output_type == 'all':
            image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))
            image_encoder_init = self.image_change(image_encoder)
            image_cls_init = self.image_cls_change(image_cls)
            image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)
        else:
            image_cls_init = self.image_cls_change(image_cls)
            image_init = image_cls_init.unsqueeze(1)

        image_mask = image_text_attention_mask[:, :image_init.size(1)] ###对于图像，所有的mask都为1
        extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())

        image_init = self.image_encoder(image_init,
                                             attention_mask=None,
                                             head_mask=None,
                                             encoder_hidden_states=None,
                                             encoder_attention_mask=extended_attention_mask,
                                             past_key_values=None,
                                             use_cache=self.use_cache,
                                             output_attentions=self.text_config.output_attentions,
                                             output_hidden_states=(self.text_config.output_hidden_states),
                                             return_dict=self.text_config.use_return_dict
                                             )
        image_init = image_init.last_hidden_state

        image_text_cat = torch.cat((image_init, text_init), dim=1) ###改为文本在后，图像在前
        # print('the shape of image_text_cat is {}'.format(image_text_cat.shape))
        # print('the shape of image_text_attention_mask is {}'.format(image_text_attention_mask.shape))

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(image_text_attention_mask, input_ids.size())

        text_image_transformer = self.text_image_encoder(image_text_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state ##(batch_size, sequence_length, hidden_size)
        # print('=========================the text_image_transformer is =========================')
        # print(text_image_transformer.shape)
        # print('the mask_pos is {}'.format(mask_pos.squeeze()))

        sequence_output = text_image_transformer
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos.squeeze()]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        # print('=========================the prediction_mask_scores is =========================')
        # print(prediction_mask_scores.shape)
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


TIMM_MODELS = {
    'nf_resnet50': 2048,
}
def is_clip_model(model_name):
    return model_name.startswith('openai/clip-')
image_model_name =  'nf_resnet50'
if image_model_name in TIMM_MODELS.keys():
    image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
elif is_clip_model(image_model_name):
    ###model_name ='openai/clip-vit-base-patch32'
    config = CLIPVisionConfig.from_pretrained(image_model_name)
    image_encoder = CLIPVisionModel.from_pretrained(
            image_model_name,
            config=config,
        )
else:
    image_encoder = AutoModel.from_pretrained(image_model_name)

# def get_image_encoder(model_name):
#     if model_name in TIMM_MODELS.keys():
#         model = timm.create_model(model_name, pretrained=True, num_classes=0)
#         # import pdb; pdb.set_trace()
#     elif is_clip_model(model_name):
#         ###model_name ='openai/clip-vit-base-patch32'
#         config = CLIPVisionConfig.from_pretrained(model_name)
#         model = CLIPVisionModel.from_pretrained(
#             model_name,
#             config=config,
#         )
#     else:
#         model = AutoModel.from_pretrained(model_name)
#     return model.cuda()

def init_image_encoder(image_model_name, frozen_image_encoder, num_image_tokens, d_text_encoder):

    # image_encoder = get_image_encoder(image_model_name)
    d_image_encoder = _d_image_encoder(image_model_name, image_encoder)

    if frozen_image_encoder:
        for p in image_encoder.parameters():
            p.requires_grad = False
            image_encoder.eval()

    proj_image_features = nn.Linear(
            in_features=d_image_encoder,
            out_features=num_image_tokens * d_text_encoder,
        )
    return proj_image_features.cuda(), d_image_encoder

def _d_image_encoder(image_model_name, image_encoder):
    ##image_model_name默认为： 'microsoft/resnet-50'
    model_name = image_model_name
    if model_name in TIMM_MODELS.keys():
        return TIMM_MODELS[model_name]
    elif is_clip_model(model_name):
        return image_encoder.config.hidden_size
    elif model_name.startswith('microsoft/resnet-'):
        return image_encoder.config.hidden_sizes[-1]
    else:
        return image_encoder.config.hidden_size

def add_image_token(model):
    num_embeddings = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(num_embeddings+1)

def reduce_image_token(model):
    num_embeddings = model.get_input_embeddings().num_embeddings
    model.resize_token_embeddings(num_embeddings-1)


def encode_images(image_encoder, proj_image_features, frozen_image_encoder, pixel_values, d_image_encoder):
    
    image_encoder = image_encoder.cuda()
    pixel_values = pixel_values.cuda()
    # print('the shape of pixel_values is {}'.format(pixel_values.shape))
    batch_size = pixel_values.shape[0]

    if frozen_image_encoder:
        with torch.no_grad():
            image_encoder.eval()
            visual = image_encoder(pixel_values)
    else:
        visual = image_encoder(pixel_values)

    if not isinstance(visual, torch.Tensor):  # HuggingFace model
        visual = visual.pooler_output

    visual = visual.reshape(batch_size, d_image_encoder)
    visual = proj_image_features(visual).cuda()
    return visual

'''bayesian_fusion_multiclass'''
# def bayesian_fusion_multiclass(all_logits, pred_classes):
#     fusion_logits =[]
#     for i in range(len(all_logits)):
#         new_preds = []
#         scores = all_logits[i]
#         for pred_class in pred_classes:
#             log_scores = torch.log(scores)
#             sum_logits = torch.sum(log_scores, axis=0)
#             exp_logits = torch.exp(sum_logits)
#             out_score = exp_logits[pred_class] / torch.sum(exp_logits)
#             new_preds.append(out_score)
#         new_preds = torch.tensor(new_preds)
#         fusion_logits.append(new_preds)
#     # print(fusion_logits)
#     fusion_logits = torch.stack(fusion_logits, 0)
#     return fusion_logits


class ResNetRobertaFinetuning(RobertaPreTrainedModel):

    def __init__(self, config, opt):
        super().__init__(config)
        print('Building the ResNetRobertaFinetuning model !!!!!!!!!!')
        self.opt = opt
        self.num_labels = config.num_labels
        self.image_model_name = opt.image_model_name ###  'microsoft/resnet-50'
        self.num_image_tokens = opt.num_image_tokens ### 2
       
        self.roberta = RobertaModel(config)
       
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()
        self.d_text_encoder = self.roberta.config.hidden_size
        
        

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None


        if self.frozen_text_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False
        # self.add_image_token() 

        

    @property
    def frozen_text_encoder(self):
        return self.opt.frozen_text_encoder
    
    @property
    def frozen_image_encoder(self):
        return self.opt.frozen_image_encoder
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        image_pixel_values_list=None,
        image_token_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        '''
        image_pixel_values_list: torch.Size([batch_size, 4, 3, 256, 256])
        4: 1+num_labels*num_demo = 1+3*1=4
        image_token_mask: (batch_size, max_text_length)
        '''
        ###for image
        # import pdb; pdb.set_trace()
        add_image_token(self.roberta)
        
        proj_image_features, d_image_encoder = init_image_encoder(self.image_model_name, self.frozen_image_encoder, self.num_image_tokens, self.d_text_encoder)
        
        
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # import pdb; pdb.set_trace()
        # print('-----------------------the input_ids shape is {}---------------'.format(input_ids.shape))
        if inputs_embeds is None:
            inputs_embeds = self.roberta.embeddings(input_ids)

        d_model = inputs_embeds.shape[-1]

        image_pixel_values_list = image_pixel_values_list.cuda().unsqueeze(1)
        
        image_features_list = []

        # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
        for pixel_values in image_pixel_values_list:
            image_features = encode_images(image_encoder=image_encoder, 
                                           proj_image_features=proj_image_features, 
                                           frozen_image_encoder=self.frozen_image_encoder, 
                                           pixel_values=pixel_values, 
                                           d_image_encoder=d_image_encoder)
            ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
            # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
            image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
            image_features_list.append(image_features)
        
        ###list[tensor]-->tensor[tensor]
        image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

        for i in range(batch_size):
            # replace <image> tokens with image_features
            if image_token_mask[i] is not None:
                ind = image_token_mask[i].nonzero(as_tuple=True)

                inputs_embeds[i][ind] = image_features_list[i].type(inputs_embeds[i].dtype)

        reduce_image_token(self.roberta)
       

        # Encode everything
        if inputs_embeds is not None:
            outputs = self.roberta(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

       
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,)
        return ((loss,) + output) if loss is not None else output


class ResNetRobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config, opt):
        super().__init__(config)
        print('Building the ResNetRobertaForPromptFinetuning model !!!!!!!!!!')
        self.opt = opt
        self.num_labels = config.num_labels
        self.image_model_name = opt.image_model_name ###  'microsoft/resnet-50'
        self.num_image_tokens = opt.num_image_tokens ### 2
       
        self.roberta = RobertaModel(config)
       
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()
        self.d_text_encoder = self.roberta.config.hidden_size
        
        

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None


        if self.frozen_text_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False
        # self.add_image_token() 

        

    @property
    def frozen_text_encoder(self):
        return self.opt.frozen_text_encoder
    
    @property
    def frozen_image_encoder(self):
        return self.opt.frozen_image_encoder

    
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        image_pixel_values_list=None,
        image_token_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_forward=False,
    ):
        '''
        image_pixel_values_list: torch.Size([batch_size, 4, 3, 256, 256])
        4: 1+num_labels*num_demo = 1+3*1=4
        image_token_mask: (batch_size, max_text_length)
        '''
        ###for image
        # import pdb; pdb.set_trace()
        add_image_token(self.roberta)
        
        proj_image_features, d_image_encoder = init_image_encoder(self.image_model_name, self.frozen_image_encoder, self.num_image_tokens, self.d_text_encoder)
        
        
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # import pdb; pdb.set_trace()
        # print('-----------------------the input_ids shape is {}---------------'.format(input_ids.shape))
        if inputs_embeds is None:
            inputs_embeds = self.roberta.embeddings(input_ids)

        d_model = inputs_embeds.shape[-1]

        image_pixel_values_list = image_pixel_values_list.cuda()
        image_features_list = []

        # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
        for pixel_values in image_pixel_values_list:
            image_features = encode_images(image_encoder=image_encoder, 
                                           proj_image_features=proj_image_features, 
                                           frozen_image_encoder=self.frozen_image_encoder, 
                                           pixel_values=pixel_values, 
                                           d_image_encoder=d_image_encoder)
            ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
            # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
            image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
            image_features_list.append(image_features)
        
        ###list[tensor]-->tensor[tensor]
        image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

        for i in range(batch_size):
            # replace <image> tokens with image_features
            if image_token_mask[i] is not None:
                ind = image_token_mask[i].nonzero(as_tuple=True)

                inputs_embeds[i][ind] = image_features_list[i].type(inputs_embeds[i].dtype)

        reduce_image_token(self.roberta)
       

        # Encode everything
        if inputs_embeds is not None:
            outputs = self.roberta(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )
        else:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

       
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output


##### added for contrastive learning
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class FusionContrastiveResNetRobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config, opt):
        super().__init__(config)
        print('Building the FusionContrastiveResNetRobertaForPromptFinetuning model !!!!!!!!!!')
        self.opt = opt
        self.num_labels = config.num_labels
        self.image_model_name = opt.image_model_name ###  'microsoft/resnet-50'
        self.num_image_tokens = opt.num_image_tokens ### 2
       
        self.roberta = RobertaModel(config)
       
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        
        self.d_text_encoder = self.roberta.config.hidden_size
        self.mlp = MLPLayer(hidden_size=self.d_text_encoder)
        self.init_weights()
        
        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None


        if self.frozen_text_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False
        # self.add_image_token() 

        

    @property
    def frozen_text_encoder(self):
        return self.opt.frozen_text_encoder
    
    @property
    def frozen_image_encoder(self):
        return self.opt.frozen_image_encoder

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        image_pixel_values_list=None,
        image_token_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_forward=False,
    ):
        '''
        input_ids: [batch_size, 2, max_length]
        attention_mask: [batch_size, 2, max_length]
        mask_pos: [batch_size, 2, 1]
        image_pixel_values_list: torch.Size([batch_size,  2, 4, 3, 256, 256]) 4: 1+num_labels*num_demo = 1+3*1=4
        image_token_mask: (batch_size, 2, max_text_length)
        '''
        ###for image
       
        
        proj_image_features, d_image_encoder = init_image_encoder(self.image_model_name, self.frozen_image_encoder, self.num_image_tokens, self.d_text_encoder)
        all_label_id =[]
        for label_id in range(len(self.label_word_list)):
            all_label_id.append(label_id)
        all_label_id = torch.tensor(all_label_id) ##for MVSA, tensor([0, 1, 2])
        batch_size = input_ids.size(0)
        
        views = input_ids.size(1)##2
        
        # print('---------------------------------the shape of input_ids is {}'.format(input_ids.shape))

        if encoder_forward is False:
            logits_list = []
            for t in range(views):
                add_image_token(self.roberta)
                input_ids_new = input_ids[:, t, :]
                # print('---------------------------------the shape of input_ids_new is {}'.format(input_ids_new.shape))
                attention_mask_new=attention_mask[:, t, :]
                mask_pos_new = mask_pos[:, t, :]
                image_pixel_values_list_new=image_pixel_values_list[:, t, :]
                image_token_mask_new=image_token_mask[:, t, :]

                if mask_pos_new is not None:
                    mask_pos_new = mask_pos_new.squeeze()

                if inputs_embeds is None:
                    inputs_embeds_new = self.roberta.embeddings(input_ids_new)
            
                d_model = inputs_embeds_new.shape[-1]

                image_pixel_values_list_new = image_pixel_values_list_new.cuda()
                image_features_list = []

                # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
                for pixel_values in image_pixel_values_list_new:
                    image_features = encode_images(image_encoder=image_encoder, 
                                                proj_image_features=proj_image_features, 
                                                frozen_image_encoder=self.frozen_image_encoder, 
                                                pixel_values=pixel_values, 
                                                d_image_encoder=d_image_encoder)
                    ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
                    # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
                    image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
                    image_features_list.append(image_features)

                    ###list[tensor]-->tensor[tensor]
                image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

                for i in range(batch_size):
                    # replace <image> tokens with image_features
                    if image_token_mask_new[i] is not None:
                        ind = image_token_mask_new[i].nonzero(as_tuple=True)

                        inputs_embeds_new[i][ind] = image_features_list[i].type(inputs_embeds_new[i].dtype)
                
                reduce_image_token(self.roberta)
                
                # Encode everything
                if inputs_embeds_new is not None:
                    outputs = self.roberta(
                        attention_mask=attention_mask_new,
                        inputs_embeds=inputs_embeds_new
                    )
                else:
                    outputs = self.roberta(
                        input_ids=input_ids_new,
                        attention_mask=attention_mask_new,
                    )
            
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos_new]

                # Logits over vocabulary tokens
                prediction_mask_scores = self.lm_head(sequence_mask_output)

                # Exit early and only return mask logits.
                if self.return_full_softmax:
                    if labels is not None:
                        return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                    return prediction_mask_scores

                # Return logits for each label
                logits = []
                for label_id in range(len(self.label_word_list)):
                    logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
                logits = torch.cat(logits, -1) ###[batch_size, num_labels]

                logits_list.append(logits) ###all_logits: [2, batch_size, num_labels]

            all_logits = torch.stack(logits_list, dim=1) ## all_logits: [batch_size, 2, num_labels]
            # print('+++++++++++++++++++++++++++all_logits.shape is {}++++++++++++++++++++++++++++++++++'.format(all_logits.shape))
            # print(all_logits)

            logits_1 = all_logits[:, 0, :]
            logits_2 = all_logits[:, 1, :]

            ##不可行，会产生负值
            # multiply_logits = all_logits[:, 0, :] * all_logits[:, 1, :] ##[btch_size, num_labels]
            # fusion_logits = multiply_logits/torch.sum(multiply_logits, dim=1).unsqueeze(dim=1) 
            
            # fusion_logits =[]
            # for i in range(len(all_logits)):
            #     new_preds = []
            #     scores = all_logits[i]
            #     for pred_class in all_label_id:
            #         log_scores = torch.log(scores)
            #         sum_logits = torch.sum(log_scores, axis=0)
            #         exp_logits = torch.exp(sum_logits)
            #         out_score = exp_logits[pred_class] / torch.sum(exp_logits)
            #         new_preds.append(out_score)
            #     new_preds = torch.tensor(new_preds)
            #     fusion_logits.append(new_preds)
            #     # print(fusion_logits)
            # fusion_logits = torch.stack(fusion_logits, 0).cuda()
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))
            # fusion_logits.requires_grad_(True)
            # print(fusion_logits.requires_grad) ##不知道为何这种写法训练有问题，模型学不到label=1的数据，换一种写法看看



            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits_1 = logsoftmax(logits_1) # Log prob of right polarity
                logits_2 = logsoftmax(logits_2) # Log prob of right polarity
                logits = (logits_1+logits_2)/2
            # import pdb; pdb.set_trace
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.KLDivLoss(log_target=True)
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss_1 = loss_fct(logits_1.view(-1, 2), labels)
                    loss_2 = loss_fct(logits_2.view(-1, 2), labels)
                    loss = (loss_1+loss_2)/2
                   
                    # loss.requirezs_grad_(True)
                    # print(loss.requires_grad)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss_1 = loss_fct(logits_1.view(-1, logits_1.size(-1)), labels.view(-1))
                    loss_2 = loss_fct(logits_2.view(-1, logits_2.size(-1)), labels.view(-1))
                    loss = (loss_1+loss_2)/2
                  
                    # print(loss.requires_grad)
                    # loss.requires_grad_(True)
                    # print(loss.requires_grad)
            # print('++++++++++++++++++++++++++loss.requires_grad is {}++++++++++++++++++++'.format(loss.requires_grad))
            # print('##################the loss is {}##########################'.format(loss))
            ###bayes fusion
            fusion_logits =[]
            for i in range(len(all_logits)):
                new_preds = []
                scores = all_logits[i]
                for pred_class in all_label_id:
                    log_scores = torch.log(scores)
                    sum_logits = torch.sum(log_scores, axis=0)
                    exp_logits = torch.exp(sum_logits)
                    out_score = exp_logits[pred_class] / torch.sum(exp_logits)
                    new_preds.append(out_score)
                new_preds = torch.tensor(new_preds)
                fusion_logits.append(new_preds)
                # print(fusion_logits)
            fusion_logits = torch.stack(fusion_logits, 0).cuda()
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))
            # fusion_logits = (logits_1+logits_2)/2 ##for ablation 
            if not fusion_logits.requires_grad:
                fusion_logits.requires_grad_(True)
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))

            output = (fusion_logits,)
            # print('~~~~~~~~~~~~~~~~~~~~~labels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(labels)
            # print('+++++++++++++++++++++++++++fusion_logits.shape is {}++++++++++++++++++++++++++++++++++'.format(fusion_logits.shape))
            # print(fusion_logits)
       
            
            if self.num_labels == 1:
                # Regression output
                output = (torch.exp(fusion_logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
            return ((loss,) + output) if loss is not None else output
        
        else:  #### encoder_forward is True
            all_mlp_output = []
            for t in range(views):
                add_image_token(self.roberta)
                input_ids_new = input_ids[:, t, :]
                attention_mask_new=attention_mask[:, t, :]
                mask_pos_new = mask_pos[:, t, :]
                image_pixel_values_list_new=image_pixel_values_list[:, t, :]
                image_token_mask_new=image_token_mask[:, t, :] 

                if mask_pos_new is not None:
                    mask_pos_new = mask_pos_new.squeeze()

                if inputs_embeds is None:
                    inputs_embeds_new = self.roberta.embeddings(input_ids_new)
            
                d_model = inputs_embeds_new.shape[-1]

                image_pixel_values_list_new = image_pixel_values_list_new.cuda()
                image_features_list = []

                # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
                for pixel_values in image_pixel_values_list_new:
                    image_features = encode_images(image_encoder=image_encoder, 
                                                proj_image_features=proj_image_features, 
                                                frozen_image_encoder=self.frozen_image_encoder, 
                                                pixel_values=pixel_values, 
                                                d_image_encoder=d_image_encoder)
                    ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
                    # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
                    image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
                    image_features_list.append(image_features)

                    ###list[tensor]-->tensor[tensor]
                image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

                for i in range(batch_size):
                    # replace <image> tokens with image_features
                    if image_token_mask_new[i] is not None:
                        ind = image_token_mask_new[i].nonzero(as_tuple=True)

                        inputs_embeds_new[i][ind] = image_features_list[i].type(inputs_embeds_new[i].dtype)
                
                reduce_image_token(self.roberta)
                # Encode everything
                if inputs_embeds_new is not None:
                    outputs = self.roberta(
                        attention_mask=attention_mask_new,
                        inputs_embeds=inputs_embeds_new
                    )
                else:
                    outputs = self.roberta(
                        input_ids=input_ids_new,
                        attention_mask=attention_mask_new,
                    )
        
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos_new]
                
                mlp_output = self.mlp(sequence_mask_output)
                # print('==================================mlp_output.shape is {}======================================'.format(mlp_output.shape))
                all_mlp_output.append(mlp_output)
            
            all_mlp_output = torch.stack(all_mlp_output, dim=1)
            # print('==================================all_mlp_output.shape is {}======================================'.format(all_mlp_output.shape))
            # return self.mlp(sequence_output[:, 0])   ### [bsz, hidden_dim]
            return all_mlp_output ##[btch_size, 2, hidden_dim]


class FusionContrastiveResNetBertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config, opt):
        super().__init__(config)
        print('Building the FusionContrastiveResNetBertForPromptFinetuning model !!!!!!!!!!')
        self.opt = opt
        self.num_labels = config.num_labels
        self.image_model_name = opt.image_model_name ###  'microsoft/resnet-50'
        self.num_image_tokens = opt.num_image_tokens ### 2
       
        self.bert = BertModel(config)
       
        self.cls = BertOnlyMLMHead(config)
        
        self.d_text_encoder = self.roberta.config.hidden_size
        self.mlp = MLPLayer(hidden_size=self.d_text_encoder)
        self.init_weights()
        
        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None


        if self.frozen_text_encoder:
            for p in self.roberta.parameters():
                p.requires_grad = False
        # self.add_image_token() 

        

    @property
    def frozen_text_encoder(self):
        return self.opt.frozen_text_encoder
    
    @property
    def frozen_image_encoder(self):
        return self.opt.frozen_image_encoder

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        image_pixel_values_list=None,
        image_token_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_forward=False,
    ):
        '''
        input_ids: [batch_size, 2, max_length]
        attention_mask: [batch_size, 2, max_length]
        mask_pos: [batch_size, 2, 1]
        image_pixel_values_list: torch.Size([batch_size,  2, 4, 3, 256, 256]) 4: 1+num_labels*num_demo = 1+3*1=4
        image_token_mask: (batch_size, 2, max_text_length)
        '''
        ###for image
       
        
        proj_image_features, d_image_encoder = init_image_encoder(self.image_model_name, self.frozen_image_encoder, self.num_image_tokens, self.d_text_encoder)
        all_label_id =[]
        for label_id in range(len(self.label_word_list)):
            all_label_id.append(label_id)
        all_label_id = torch.tensor(all_label_id) ##for MVSA, tensor([0, 1, 2])
        batch_size = input_ids.size(0)
        
        views = input_ids.size(1)##2
        
        # print('---------------------------------the shape of input_ids is {}'.format(input_ids.shape))

        if encoder_forward is False:
            logits_list = []
            for t in range(views):
                add_image_token(self.roberta)
                input_ids_new = input_ids[:, t, :]
                # print('---------------------------------the shape of input_ids_new is {}'.format(input_ids_new.shape))
                attention_mask_new=attention_mask[:, t, :]
                token_type_ids_new = token_type_ids[:, t, :]
                mask_pos_new = mask_pos[:, t, :]
                image_pixel_values_list_new=image_pixel_values_list[:, t, :]
                image_token_mask_new=image_token_mask[:, t, :]

                if mask_pos_new is not None:
                    mask_pos_new = mask_pos_new.squeeze()

                if inputs_embeds is None:
                    inputs_embeds_new = self.roberta.embeddings(input_ids_new)
            
                d_model = inputs_embeds_new.shape[-1]

                image_pixel_values_list_new = image_pixel_values_list_new.cuda()
                image_features_list = []

                # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
                for pixel_values in image_pixel_values_list_new:
                    image_features = encode_images(image_encoder=image_encoder, 
                                                proj_image_features=proj_image_features, 
                                                frozen_image_encoder=self.frozen_image_encoder, 
                                                pixel_values=pixel_values, 
                                                d_image_encoder=d_image_encoder)
                    ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
                    # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
                    image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
                    image_features_list.append(image_features)

                    ###list[tensor]-->tensor[tensor]
                image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

                for i in range(batch_size):
                    # replace <image> tokens with image_features
                    if image_token_mask_new[i] is not None:
                        ind = image_token_mask_new[i].nonzero(as_tuple=True)

                        inputs_embeds_new[i][ind] = image_features_list[i].type(inputs_embeds_new[i].dtype)
                
                reduce_image_token(self.roberta)
                
                # Encode everything
                if inputs_embeds_new is not None:
                    outputs = self.bert(
                        attention_mask=attention_mask_new,
                        inputs_embeds=inputs_embeds_new,
                        token_type_ids=token_type_ids_new
                    )
                else:
                    outputs = self.bert(
                        input_ids=input_ids_new,
                        attention_mask=attention_mask_new,
                        token_type_ids=token_type_ids_new
                    )
            
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos_new]

                # Logits over vocabulary tokens
                prediction_mask_scores = self.lm_head(sequence_mask_output)

                # Exit early and only return mask logits.
                if self.return_full_softmax:
                    if labels is not None:
                        return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                    return prediction_mask_scores

                # Return logits for each label
                logits = []
                for label_id in range(len(self.label_word_list)):
                    logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
                logits = torch.cat(logits, -1) ###[batch_size, num_labels]

                logits_list.append(logits) ###all_logits: [2, batch_size, num_labels]

            all_logits = torch.stack(logits_list, dim=1) ## all_logits: [batch_size, 2, num_labels]
            # print('+++++++++++++++++++++++++++all_logits.shape is {}++++++++++++++++++++++++++++++++++'.format(all_logits.shape))
            # print(all_logits)

            logits_1 = all_logits[:, 0, :]
            logits_2 = all_logits[:, 1, :]

            ##不可行，会产生负值
            # multiply_logits = all_logits[:, 0, :] * all_logits[:, 1, :] ##[btch_size, num_labels]
            # fusion_logits = multiply_logits/torch.sum(multiply_logits, dim=1).unsqueeze(dim=1) 
            
            # fusion_logits =[]
            # for i in range(len(all_logits)):
            #     new_preds = []
            #     scores = all_logits[i]
            #     for pred_class in all_label_id:
            #         log_scores = torch.log(scores)
            #         sum_logits = torch.sum(log_scores, axis=0)
            #         exp_logits = torch.exp(sum_logits)
            #         out_score = exp_logits[pred_class] / torch.sum(exp_logits)
            #         new_preds.append(out_score)
            #     new_preds = torch.tensor(new_preds)
            #     fusion_logits.append(new_preds)
            #     # print(fusion_logits)
            # fusion_logits = torch.stack(fusion_logits, 0).cuda()
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))
            # fusion_logits.requires_grad_(True)
            # print(fusion_logits.requires_grad) ##不知道为何这种写法训练有问题，模型学不到label=1的数据，换一种写法看看



            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits_1 = logsoftmax(logits_1) # Log prob of right polarity
                logits_2 = logsoftmax(logits_2) # Log prob of right polarity
                logits = (logits_1+logits_2)/2
            # import pdb; pdb.set_trace
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.KLDivLoss(log_target=True)
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss_1 = loss_fct(logits_1.view(-1, 2), labels)
                    loss_2 = loss_fct(logits_2.view(-1, 2), labels)
                    loss = (loss_1+loss_2)/2
                   
                    # loss.requirezs_grad_(True)
                    # print(loss.requires_grad)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss_1 = loss_fct(logits_1.view(-1, logits_1.size(-1)), labels.view(-1))
                    loss_2 = loss_fct(logits_2.view(-1, logits_2.size(-1)), labels.view(-1))
                    loss = (loss_1+loss_2)/2
                  
                    # print(loss.requires_grad)
                    # loss.requires_grad_(True)
                    # print(loss.requires_grad)
            # print('++++++++++++++++++++++++++loss.requires_grad is {}++++++++++++++++++++'.format(loss.requires_grad))
            # print('##################the loss is {}##########################'.format(loss))
            ###bayes fusion
            fusion_logits =[]
            for i in range(len(all_logits)):
                new_preds = []
                scores = all_logits[i]
                for pred_class in all_label_id:
                    log_scores = torch.log(scores)
                    sum_logits = torch.sum(log_scores, axis=0)
                    exp_logits = torch.exp(sum_logits)
                    out_score = exp_logits[pred_class] / torch.sum(exp_logits)
                    new_preds.append(out_score)
                new_preds = torch.tensor(new_preds)
                fusion_logits.append(new_preds)
                # print(fusion_logits)
            fusion_logits = torch.stack(fusion_logits, 0).cuda()
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))
            if not fusion_logits.requires_grad:
                fusion_logits.requires_grad_(True)
            # print('------------------------fusion_logits.requires_grad is {}-------------'.format(fusion_logits.requires_grad))

            output = (fusion_logits,)
            # print('~~~~~~~~~~~~~~~~~~~~~labels~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            # print(labels)
            # print('+++++++++++++++++++++++++++fusion_logits.shape is {}++++++++++++++++++++++++++++++++++'.format(fusion_logits.shape))
            # print(fusion_logits)
       
            
            if self.num_labels == 1:
                # Regression output
                output = (torch.exp(fusion_logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
            return ((loss,) + output) if loss is not None else output
        
        else:  #### encoder_forward is True
            all_mlp_output = []
            for t in range(views):
                add_image_token(self.roberta)
                input_ids_new = input_ids[:, t, :]
                token_type_ids_new = token_type_ids[:, t, :]
                attention_mask_new=attention_mask[:, t, :]
                mask_pos_new = mask_pos[:, t, :]
                image_pixel_values_list_new=image_pixel_values_list[:, t, :]
                image_token_mask_new=image_token_mask[:, t, :] 

                if mask_pos_new is not None:
                    mask_pos_new = mask_pos_new.squeeze()

                if inputs_embeds is None:
                    inputs_embeds_new = self.roberta.embeddings(input_ids_new)
            
                d_model = inputs_embeds_new.shape[-1]

                image_pixel_values_list_new = image_pixel_values_list_new.cuda()
                image_features_list = []

                # print('======================================the shape of image_pixel_values_list is {}======================================'.format(image_pixel_values_list.shape))
                for pixel_values in image_pixel_values_list_new:
                    image_features = encode_images(image_encoder=image_encoder, 
                                                proj_image_features=proj_image_features, 
                                                frozen_image_encoder=self.frozen_image_encoder, 
                                                pixel_values=pixel_values, 
                                                d_image_encoder=d_image_encoder)
                    ###image_features: (image_token_mask, num_image_tokens*1024) (4, 2048)
                    # print("======================================the shape of image_features is {}=====================================".format(image_features.shape))
                    image_features = image_features.reshape(-1, d_model) ###(image_token_mask*num_image_tokens, 1024) (4*2, 1024)
                    image_features_list.append(image_features)

                    ###list[tensor]-->tensor[tensor]
                image_features_list = torch.stack(image_features_list) ###(batch_size, 8, 1024)

                for i in range(batch_size):
                    # replace <image> tokens with image_features
                    if image_token_mask_new[i] is not None:
                        ind = image_token_mask_new[i].nonzero(as_tuple=True)

                        inputs_embeds_new[i][ind] = image_features_list[i].type(inputs_embeds_new[i].dtype)
                
                reduce_image_token(self.roberta)
                # Encode everything
                if inputs_embeds_new is not None:
                    outputs = self.roberta(
                        attention_mask=attention_mask_new,
                        inputs_embeds=inputs_embeds_new,
                        token_type_ids=token_type_ids_new,
                    )
                else:
                    outputs = self.roberta(
                        input_ids=input_ids_new,
                        attention_mask=attention_mask_new,
                        token_type_ids=token_type_ids_new,
                    )
        
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos_new]
                
                mlp_output = self.mlp(sequence_mask_output)
                # print('==================================mlp_output.shape is {}======================================'.format(mlp_output.shape))
                all_mlp_output.append(mlp_output)
            
            all_mlp_output = torch.stack(all_mlp_output, dim=1)
            # print('==================================all_mlp_output.shape is {}======================================'.format(all_mlp_output.shape))
            # return self.mlp(sequence_output[:, 0])   ### [bsz, hidden_dim]
            return all_mlp_output ##[btch_size, 2, hidden_dim]

class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output

class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output
