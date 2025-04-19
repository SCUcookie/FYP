#!/bin/bash
# 第一个命令：显示当前目录
lm_eval --model hf \
    --model_args pretrained=save/llama/magnitude/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama/magnitude/ \

lm_eval --model hf \
    --model_args pretrained=save/llama/ours/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama/ours/ \

lm_eval --model hf \
    --model_args pretrained=save/llama/wanda/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama/wanda/ \

lm_eval --model hf \
    --model_args pretrained=save/llama/sparse/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama/sparse/ \

lm_eval --model hf \
    --model_args pretrained=save/llama2/magnitude/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama2/magnitude/ \

lm_eval --model hf \
    --model_args pretrained=save/llama2/ours/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama2/ours/ \

lm_eval --model hf \
    --model_args pretrained=save/llama2/wanda/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama2/wanda/ \

lm_eval --model hf \
    --model_args pretrained=save/llama2/sparse/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama2/sparse/ \

lm_eval --model hf \
    --model_args pretrained=save/opt/magnitude/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/opt/magnitude/ \

lm_eval --model hf \
    --model_args pretrained=save/opt/ours/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/opt/ours/ \

lm_eval --model hf \
    --model_args pretrained=save/opt/wanda/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/opt/wanda/ \

lm_eval --model hf \
    --model_args pretrained=save/opt/sparse/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/opt/sparse/ \