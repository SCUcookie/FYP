import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *
import bisect

from result.llama2_7B import hessian_trace

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def get_llama2(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

dev=torch.device('cuda')

def compute_global_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只统计权重矩阵的稀疏度
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()
    
    global_sparsity = zero_params / total_params * 100
    return global_sparsity

@torch.no_grad()
def llama2_sequential(model, dataloader, dev, method="pruning", sparsity_way="origin", sensitivity=None, total_weight=None):
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev) 
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    ###经过embed层之后捕获第一层的输入
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

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')
    print("Pruning ...")

    sensitivity = sensitivity
    total_weight = total_weight

    first,third=0,0
    id_list=[]

    clayer = 0
    sen = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}
            gpts = {}

            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                position_ids = torch.arange(0, model.seqlen, dtype=torch.long, device=dev).unsqueeze(0)
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            def get_uniform_sparsity(sen):
                def get_step(a, b, s, n):
                    d = 2*(s-a*n) / (n*(n-1))
                    if d <= (b-a)/(n-1):
                        return d
                    else:
                        return 0
                low_bound = .4
                upper_bound = 1.0
                lens = len(layers)
                spa = args.sparsity
                last_layer_num = 0
                sparsity = (lens * spa - last_layer_num) / (lens - last_layer_num)
                if last_layer_num != 0:
                    sen = sen[:-last_layer_num]
                num_layer = lens - last_layer_num
                d = get_step(low_bound, upper_bound, sparsity * num_layer, num_layer)
                _, id = torch.sort(sen, dim=-1)
                sen[id] = torch.arange(low_bound, low_bound + num_layer*d, d)[:num_layer]
                sen = torch.cat((sen, torch.ones(last_layer_num)), dim=-1)
                return sen
                
            def get_layer_sparsity(l):
                sen = torch.as_tensor(sensitivity)
                last_layer_num = 1
                sparsity = (len(layers) * args.sparsity - last_layer_num) / (len(layers) - last_layer_num)
                sen = sen[:-last_layer_num]
                num_layer = len(layers) - last_layer_num
                normalize_sen = sen / sen.sum()
                sen = normalize_sen * num_layer * sparsity
                while torch.any(sen>1.0).item():
                    sen = torch.softmax(sen, dim=-1) * num_layer * sparsity
                sen = torch.cat((sen, torch.ones(last_layer_num)), dim=-1)
                return 1 - sen[l]
                
            def get_weight_sparsity(layer, name):
                id = bisect.bisect_left(total_weight, sensitivity[layer][name]) 
                if id == len(total_weight):
                    id = len(total_weight) - 1
                id_list.append(id)
                lower_bound = 0.35
                upper_bound = 2 * args.sparsity - lower_bound
                sen = lower_bound + id * (upper_bound - lower_bound) / (len(total_weight) - 1 )
                return 1-sen 
        
            for name in gpts:
                print(i, name)
                if method == "pruning":
                    if sparsity_way == "origin":
                        sparsity = args.sparsity
                    elif sparsity_way == "layer-level":
                        sparsity = get_layer_sparsity(i)
                    elif sparsity_way == "weight-level":
                        sparsity = get_weight_sparsity(i, name)
                        # print(sparsity)
                        if sparsity<=0.4:
                            prunen=3
                            first+=1
                        elif sparsity<=0.6:
                            prunen=4
                        else:
                            prunen=5
                            third+=1
                        prunem=8
                    gpts[name].fasterprune(
                        sparsity,
                        prunen=args.prunen,
                        prunem=args.prunem,
                        percdamp=args.percdamp,
                        blocksize=args.blocksize,
                    )
                gpts[name].free()

        for j in range(args.nsamples):
            position_ids = torch.arange(0, model.seqlen, dtype=torch.long, device=dev).unsqueeze(0)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    print(first,third)
    print(f"id_list:{id_list} len_id:{len(id_list)}")
    print(f"len_totalweight:{len(total_weight)}")
    global_sparsity = compute_global_sparsity(model)
    print(f"global sparsity: {global_sparsity}")
    model.config.use_cache = use_cache

@torch.no_grad()
def llama2_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}
 
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
    
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            position_ids = torch.arange(0, model.seqlen, dtype=torch.long, device=dev).unsqueeze(0)
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()  ### eos remove
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:] ### sos remove
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache

def get_sensitivity():
    sensitivity = []
    sen = [0]*len(model.model.layers)
    dict = {}
    clayer = 0 
    total_weight = []
    for name, trace in hessian_trace.items():
        if name.startswith("model.layers"):
            layer = int(name.split(".")[2])
            if clayer < layer:
                clayer = layer
                sensitivity.append(dict)
                dict = {}
            subname = ".".join(name.split(".")[3:])
            if subname.endswith(".weight"):
                dict[subname[:-7]] = trace
                total_weight.append(trace)
                sen[layer] += trace
    sensitivity.append(dict)
    total_weight = sorted(total_weight)
    return sensitivity, total_weight

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, 
        help='LlaMA model to load'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
        "--sparsity_way", type=str, default="origin", help="Sparsity way"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local_rank"
    )
    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama2(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    dataloader = [(inputs.to(dev), targets.to(dev)) for inputs, targets in dataloader]
    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        sensitivity, total_weight = get_sensitivity()
        llama2_sequential(model, dataloader, dev, sparsity_way=args.sparsity_way, sensitivity=sensitivity,total_weight=total_weight)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)
    
    datalist = ['wikitext2', 'ptb', 'c4']
    for dataset in datalist:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama2_eval(model, testloader, dev, dataset, args.log_wandb)

    if args.save:
        model.save_pretrained(args.save)