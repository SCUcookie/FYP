# Final Year Project: mix semi-structured pruning method(2:4,4:8) to prune LLMs

**Sensitivity-Guided Dynamic Hybrid Sparse Pruning for Large Language Models**

## Usage

Below is the command for running pruning by using opt model:
```sh
run the prune.py: python prune.py --model opt/125m --dataset c4 --sparsity .5 --sparsity_way weight-level(layer-level)
```
For llama model:
```sh
run the prune.py: python llama.py --model model/llama-7b --dataset c4 --sparsity.5 --sparsity_way weight-level(layer-level)
```
For llama2 model:
```sh
run the prune.py: python llama.py --model model/llama-2-7b-hf --dataset c4 --sparsity.5 --sparsity_way weight-level(layer-level)
```
## Zero-shot Evaluation

Run the single command:
```sh
lm_eval --model hf \
    --model_args pretrained=save/llama/magnitude/ \
    --tasks truthfulqa,hellaswag,arc_challenge,piqa \
    --device cuda:0 \
    --batch_size auto:1 \
    --output_path ./eval_out/llama/magnitude/ \
```

Run the whole process:
```sh
sh eval.sh
```