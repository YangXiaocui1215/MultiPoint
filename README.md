# Few-shot Multimodal Sentiment Analysis based on Multimodal Probabilistic Fusion Prompts

Thanks for your stay in this repo. This project aims to multimodal sentiment detection in the few-shot setting. The paper can be found in [here](https://arxiv.org/abs/2211.06607).

## Data

You can download data from the [Google Drive](https://drive.google.com/file/d/1gsXi0x-rY1YOnDDWVJIjSWSFRiQtzRoE/view?usp=drive_link).

## For text modality:
We employ roberta-large to text, roberta-large can also be replaced by bert-base, bert-large, roberta-base and distilbert-base (see [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for details).
## For image modality:
We employ NF-Resnet50, you can download the pretrained weights from [https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/nf_resnet50_ra2-9f236009.pth].

# Experiments with multiple runs

To carry out experiments with multiple data splits, as the evaluation protocol detailed in our paper (grid-search for each seed and 5 different seeds), you can use the following scripts:

## For MVSA_Single:

``` python
for seed in 13 21 42 87 100 #### random seeds for different train-test splits
do
    for bs in 8   #### batch size for each GPU
    do
        for lr in 8e-6 #### learning rate for MLM loss 
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=LM-BFF-Fusion-Add-Caption \
                TYPE=prompt-demo \
                TASK=Mul_MVSA_Single_Fusion_Add_Caption \
                BS=$bs \
                LR=$lr \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_experiment_fusion_add_caption_for_MVSA_Single.sh "--max_seq_length 256 --demo_filter --demo_filter_model sbert-roberta-large --num_sample 4"
            done
        done
    done
done
```

## For MVSA_Multiple:

``` python
for seed in 13 21 42 87 100 #### random seeds for different train-test splits
do
    for bs in 8   #### batch size for each GPU
    do
        for lr in 8e-6 #### learning rate for MLM loss
        do
            for supcon_lr in 1e-5    #### learning rate for SupCon loss
            do
                TAG=LM-BFF-Fusion-Add-Caption \
                TYPE=prompt-demo \
                TASK=Mul_MVSA_Multiple_Fusion_Add_Caption \
                BS=$bs \
                LR=$lr \
                SEED=$seed \
                MODEL=roberta-large \
                bash run_experiment_fusion_add_caption_for_MVSA_Multiple.sh "--max_seq_length 256 --demo_filter --demo_filter_model sbert-roberta-large --num_sample 4"
            done
        done
    done
done
```
