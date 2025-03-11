import time
import torch
import torch.nn as nn
from quant import *
from sparsegpt import *
from modelutils import *
import bisect

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def device_collate_fn(batch):
    inputs, targets = batch
    return inputs.to(dev), targets.to(dev)

dev=torch.device('cuda')

@torch.no_grad()
def get_sensitivity(model, dataloader, dev, sparsity_way="origin", args=None):
    print('Starting Sensitivity Calculation...')
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)
    layers = model.model.decoder.layers
    # Move initial layers to device
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    # Initialize containers
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    # Capture input data
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Initialize sensitivity containers
    if sparsity_way == "layer-level":
        sensitivity = [0.0] * len(layers)
    elif sparsity_way == "weight-level":
        sensitivity = []
        total_weight = []
    else:
        raise ValueError("Invalid sparsity_way")

    # Main sensitivity calculation loop
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        # Prepare for sensitivity calculation
        from Myhessian import Hessian as hessian  # Local import to avoid dependency when not needed
        from convert import precision_context
        # Enable gradient for sensitivity calculation
        with torch.enable_grad():
            original_dtypes = {name: param.dtype for name, param in model.named_parameters()}
            dataloader = dataloader[:min(1, len(dataloader))]
            model.to(dev).train()  # Switch to train mode for gradient computation
            with precision_context(model, torch.float32):
                hes = hessian(model, nn.CrossEntropyLoss(), dataloader=dataloader)
                hessian_trace = hes.trace()
            for name, param in model.named_parameters():
                param.data = param.data.to(original_dtypes[name])
            print(hessian_trace)
            # Update sensitivity based on trace results
            if sparsity_way == "layer-level":
                for name, trace in hessian_trace.items():
                    if name.startswith("model.decoder.layers"):
                        layer_idx = int(name.split('.')[3])
                        sensitivity[layer_idx] += trace
            elif sparsity_way == "weight-level":
                layer_dict = {}
                for name, trace in hessian_trace.items():
                    if name.startswith(f"model.decoder.layers.{i}"):
                        weight_name = '.'.join(name.split('.')[4:-1])  # Extract weight name
                        if 'weight' in name:
                            layer_dict[weight_name] = trace
                            total_weight.append(trace)
                sensitivity.append(layer_dict)

        # Clean up
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    # Post-processing
    if sparsity_way == "weight-level":
        total_weight = sorted(total_weight)
        return sensitivity, total_weight
    else:
        return sensitivity, None

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='OPT model to load; pass `facebook/opt-X`.')
    parser.add_argument('--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'], default='wikitext2', help='Calibration dataset')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--sparsity_way', type=str, choices=['layer-level', 'weight-level'], default='layer-level', 
                       help='Sensitivity calculation granularity')
    parser.add_argument('--save_sensitivity', type=str, help='Path to save sensitivity results')
    args = parser.parse_args()

    # Initialize model
    model = get_opt(args.model)
    model.eval()

    # Load data
    dataloader, _ = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=0,  # Fixed seed for reproducibility
        model=args.model, 
        seqlen=model.seqlen
    )

    dataloader = [(inputs.to(dev), targets.to(dev)) for inputs, targets in dataloader]

    # Calculate sensitivity
    sensitivity, total_weight = get_sensitivity(
        model, 
        dataloader, 
        dev=torch.device('cuda'), 
        sparsity_way=args.sparsity_way
    )

    # Save results
    if args.save_sensitivity:
        results = {
            'sensitivity': sensitivity,
            'total_weight': total_weight,
            'metadata': {
                'model': args.model,
                'dataset': args.dataset,
                'nsamples': args.nsamples,
                'sparsity_way': args.sparsity_way
            }
        }
        torch.save(results, args.save_sensitivity)
        print(f"Sensitivity results saved to {args.save_sensitivity}")

    # Print summary
    print("\n=== Sensitivity Summary ===")
    if args.sparsity_way == "layer-level":
        for idx, sens in enumerate(sensitivity):
            print(f"Layer {idx:3d} | Sensitivity: {sens:.4f}")
    else:
        print("Weight-level sensitivities stored in nested structure")