# Few-shot Multimodal Sentiment Analysis based on Multimodal Probabilistic Fusion Prompts

Thanks for your stay in this repo. This project aims to multimodal sentiment detection in the few-shot setting. The paper can be found in [here](https://arxiv.org/abs/2211.06607).

# Experiments with multiple runs

To carry out experiments with multiple data splits, as the evaluation protocol detailed in our paper (grid-search for each seed and 5 different seeds), you can use the following scripts:

## For MVSA_Single:

```
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

```
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
