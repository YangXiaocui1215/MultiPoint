# Required environment variables:
# TAG: tag for the trail
# TYPE: finetune / prompt / prompt-demo  
# TASK: SST-2 / sst-5 / mr / cr / mpqa / subj / trec / CoLA / MNLI / SNLI / QNLI / RTE / MRPC / QQP / STS-B
# BS: batch size (recommendation: 2 / 4 / 8)
# LR: learning rate (recommendation: 1e-5 / 2e-5 / 5e-5)
# SEED: random seed (13 / 21 / 42 / 87 / 100)
# MODEL: pre-trained model name (roberta-*, bert-*), see Transformers model list

# Number of training instances per label
K=32

# Training steps
MAX_STEP=500

# Validation steps
EVAL_STEP=50

TEXT_MAX_LENGTH=256
ADD_IMAGE=True
EVALUATE_DURING_TRAINING=True
IMAGE_MODEL_NAME='nf_resnet50'
SupCon_LR=1e-5
# Task specific parameters
# The default length is 128 and the default number of samples is 16.
# For some tasks, we use longer length or double demo (when using demonstrations, double the maximum length).
# For some tasks, we use smaller number of samples to save time (because of the large size of the test sets).
# All those parameters are set arbitrarily by observing the data distributions.
TASK_EXTRA=""
NUM_IMAGE_TOKENS=2
IMAGE_TOKEN="<image>_0"

case $TASK in
    Mul_MVSA_Single_Fusion_Add_Caption)
        TEMPLATE=*cls*$IMAGE_TOKEN*is*caption_0*sep+**sent_0*_It_was*mask*.*sep+* 
        TEMPLATE2=*cls*$IMAGE_TOKEN*is*caption_0*sep+*_The_sentense_\"*sent_0*\"_has*mask*_sentiment*sep+*
        MAPPING="{'negative':'terrible','neutral':'okay','positive':'great'}"
        TASK_EXTRA="--max_seq_len 256 --num_sample 4"
        ;;
    Mul_MVSA_Multiple_Fusion_Add_Caption)
        TEMPLATE=*cls*$IMAGE_TOKEN*is*caption_0*sep+**sent_0*_It_was*mask*.*sep+*
        TEMPLATE2=*cls*$IMAGE_TOKEN*is*caption_0*sep+*_The_sentense_\"*sent_0*\"_has*mask*_sentiment*sep+* 
        MAPPING="{'negative':'terrible','neutral':'okay','positive':'great'}"
        ;;
esac

# Gradient accumulation steps
# For medium-sized GPUs (e.g., 2080ti with 10GB memory), they can only take 
# a maximum batch size of 2 when using large-size models. So we use gradient
# accumulation steps to achieve the same effect of larger batch sizes.
REAL_BS=2
# GS=$(expr $BS / $REAL_BS)
GS=1

# Use a random number to distinguish different trails (avoid accidental overwriting)
TRIAL_IDTF=$RANDOM
DATA_DIR=data/k-shot/MVSA_Single/$K-$SEED

CUDA_VISIBLE_DEVICES=3 python run_new_add_caption.py \
  --task_name $TASK \
  --data_dir $DATA_DIR \
  --overwrite_output_dir \
  --add_image $ADD_IMAGE \
  --image_model_name $IMAGE_MODEL_NAME \
  --num_image_tokens $NUM_IMAGE_TOKENS \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --model_name_or_path $MODEL \
  --few_shot_type $TYPE \
  --num_k $K \
  --max_seq_length $TEXT_MAX_LENGTH \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps $GS \
  --learning_rate $LR \
  --max_steps $MAX_STEP \
  --logging_steps $EVAL_STEP \
  --eval_steps $EVAL_STEP \
  --num_train_epochs 0 \
  --output_dir output/result_fusion_add_caption/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
  --seed $SEED \
  --tag $TAG \
  --template $TEMPLATE \
  --template2 $TEMPLATE2 \
  --mapping $MAPPING \
  --template_list_location my_auto_template/MVSA_Single/32-$SEED.sort.txt \
  --supcon_learning_rate $SupCon_LR \
  $TASK_EXTRA \
  $1 

# Delete the checkpoint 
# Since we need to run multiple trials, saving all the checkpoints takes 
# a lot of storage space. You can find all evaluation results in `log` file anyway.
rm -r output/result_fusion_add_caption/$TASK-$TYPE-$K-$SEED-$MODEL-$TRIAL_IDTF \
