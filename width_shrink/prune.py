import time 
import heapq 
import torch 
import torch.nn as nn 
from .data import get_loaders 
import bitsandbytes as bnb

from peft.tuners.lora.bnb  import Linear4bit, Linear8bitLt
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from peft.utils.integrations import dequantize_bnb_weight

import numpy as np
import re
import sys

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        # self.rows = layer.weight.data.shape[0]
        # self.columns = layer.weight.data.shape[1]

        self.rows = layer.out_features
        self.columns = layer.in_features

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        # print(inp.shape)  # [1, seq_len, hidden_size]
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Linear4bit) or isinstance(self.layer, Linear8bitLt) or isinstance(self.layer, bnb.nn.Linear4bit) or isinstance(self.layer, bnb.nn.Linear8bitLt):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # [hidden_size, seq_len]

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp
        
        # print(inp.shape)
        # print(self.scaler_row.shape)

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
        
def find_layers(module, layers=[nn.Linear, Linear4bit, Linear8bitLt, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        if 'lora' in name:
            continue
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.model.model.layers
        
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device, nsamples=128, seqlen=2048):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = module.self_attn
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache 
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.num_calib_sample, seed=42, tokenizer=tokenizer)
    print("dataset loading complete")
    
    activation_score = {}
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.num_calib_sample)

        if hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            layers = model.model.model.layers
            
        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            # print(f'All modules in layer {i}:', subset.keys())
            # input()
            
            if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(subset[name], layer_name=name)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.num_calib_sample):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"pruning layer {i} name {name}")

                # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))  ## [out_features, in_features] * [1, in_features]
                
                activation_score[f'model.layers.{i}.{name}'] = torch.sqrt(wrapped_layers[name].scaler_row) # .cpu().numpy()  # [in_features]
                
                # print(name)
                # print(W_metric.shape)
                # print(subset[name].weight.shape)
                # print(subset[name].weight.dtype)
                # input()
                
                # W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

                # sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                # W_mask.scatter_(1, indices, True)


            for j in range(args.num_calib_sample):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    
    device_map = 'cpu' # list(activation_score.values())[0].device
    shrink_config = {'enable_shrinking': args.enable_shrinking, 
                    "shrinkable_width": args.shrinkable_width,
                    "shrinking_method": args.shrinking_method,
                    "shrinking_file": args.shrinking_file,
                    "mask_dtype": "torch.float16" if args.fp16 else ("torch.bfloat16" if args.bf16 else "torch.float32")}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        shrink_config=shrink_config
    )
    
    score = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            print('Calcuting the score for: ', name)
            
            # print(module.weight.data.dtype)
            # print(module.weight.device)
            # print(activation_score[name].device)
            
            with torch.no_grad():
                W_metric = module.weight.data.to(activation_score[name]).abs() * activation_score[name].reshape(1, -1)  # [out_features, in_features]
                
                if args.prune_width_dim == 'out':
                    W_metric = W_metric.sum(-1).cpu().numpy()  # [out_features], indicating the importance of each output neuron
                else:
                    W_metric = W_metric.sum(0).cpu().numpy()  # [in_features], indicating the importance of each output neuron
            
            # print(W_metric.shape, out_W_metric.shape)
            
            score[name] = W_metric
    
    mask = {}
    
    ratios = [1, 7/8, 3/4, 5/8, 1/2]  # [7/8, 3/4, 5/8, 1/2, 3/8, 1/4]
    for name, s in score.items():
        mask[name] = {}
        for ratio in ratios:
            num_ch = int(len(s) * ratio)
            mask[name][ratio] = np.zeros_like(s)
            mask[name][ratio][np.argsort(s)[::-1][:num_ch]] = 1
                
    return mask



# Define BiasGPT class
class BiasGPT:
    """
    This class wraps a GPT layer for specific operations.
    """
    def __init__(self, layer, metric):
        self.layer = layer
        self.dev = self.layer.weight.device
        # self.out_dim = layer.weight.data.shape[0]
        # self.in_dim = layer.weight.data.shape[1]

        self.out_dim = layer.out_features
        self.in_dim = layer.in_features
        
        self.type = metric
        self.nsamples = 0
        
        self.baseline_inp = torch.zeros((self.in_dim), device=self.dev)
        if self.type == "WIFN":
            self.scaler_inp = torch.zeros((self.in_dim), device=self.dev)
        else:   
            self.fluc_inp = torch.zeros((self.in_dim), device=self.dev)

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, Linear4bit) or isinstance(self.layer, Linear8bitLt):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()   # (dim, seqlen)

        old_baseline_inp = self.baseline_inp
        self.baseline_inp *= self.nsamples / (self.nsamples + batch_size)
        self.baseline_inp += torch.mean(inp, dim=1) / (self.nsamples + batch_size)
        if self.type == "WIFN":
            inp = inp.type(torch.float32)
            self.scaler_inp *= self.nsamples / (self.nsamples + batch_size)
            self.scaler_inp += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + batch_size)
        else:
            if self.nsamples == 0:
                self.fluc_inp = 0
            else:
                self.fluc_inp *= (self.nsamples - 1) / (self.nsamples + batch_size - 1)
                self.fluc_inp += torch.sum((inp - self.baseline_inp.unsqueeze(1)) * (inp - old_baseline_inp.unsqueeze(1)), dim=1) / (self.nsamples + batch_size)   # a²+b²+c²...没开根号

        self.nsamples += batch_size

        
    def free(self):
        self.baseline_inp = None
        if hasattr(self, 'fluc_inp'):
            self.fluc_inp = None
        if hasattr(self, 'scaler_inp'):
            self.scaler_inp = None
        torch.cuda.empty_cache()  
        
        
def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))



def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    
    args.seqlen = 512
    
    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.num_calib_sample, seed=42, tokenizer=tokenizer, seqlen=args.seqlen)
    print("dataset loading complete")

    def dequantize(layer):
        weight = layer.get_base_layer().weight
        dequant_weight = dequantize_bnb_weight(weight, state=weight.quant_state)
        
        return dequant_weight
    
    metrics = {
        'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
        'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(dequantize(subset[name]).data.pow(2), dim=0),
        'WIFN': lambda wrapped_layers, subset, name: (torch.abs(dequantize(subset[name]).data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
    }
    
    args.metrics = 'WIFV'
    args.structure = 'AL-AM'

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, nsamples=args.num_calib_sample, seqlen=args.seqlen)
    
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.model.model.layers

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []
    
    # Split into sub-problems, separate statistics for each module
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics) if not args.wanda_sp else WrappedGPT(subset[name])     

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.num_calib_sample):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == 'self_attn.o_proj':
                if not args.wanda_sp:
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                    baseline_inp = wrapped_layers[name].baseline_inp.type(torch.half)
                else:
                    W_metric = (torch.abs(dequantize(subset[name]).data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))).mean(axis=0)
                    baseline_inp = wrapped_layers[name].scaler_row.type(torch.half)
                    
                if args.structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:  ## always use this branch
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(baseline_inp)
            else:
                if not args.wanda_sp:
                    W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                    baseline_inp = wrapped_layers[name].baseline_inp.type(torch.half)
                else:
                    W_metric = (torch.abs(dequantize(subset[name]).data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))).mean(axis=0)
                    baseline_inp = wrapped_layers[name].scaler_row.type(torch.half)
                    
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.pruning_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:  ## always use this branch
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(baseline_inp)
            
            if hasattr(wrapped_layers[name], 'free'):
                wrapped_layers[name].free()

        inps, outs = outs, inps # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
    
    mask = {}
    bias = {}
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) or isinstance(module, Linear4bit) or isinstance(module, Linear8bitLt)) and 'lm_head' not in name and 'lora' not in name and 'base_layer' not in name:
            mask[name] = {}
            bias[name] = {}
    
    # hkolisetty6
    pruning_ratio_list = [1, 7/8, 3/4, 5/8, 1/2]
                
    for pruning_ratio in pruning_ratio_list:  ## meaning the remaining ratio
        if args.structure in ["AL-MM", "AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
            
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)
            
            if args.structure == "AL-MM":
                sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
                attn_thres = sorted_attn[-int(args.remove_heads)]
                attn_mask = (attn_metric > attn_thres)  # 1 means retain
                
                sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
                mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
                mlp_mask = (mlp_metric > mlp_thres)
            else:  ## always use this branch
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(indices)
                compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*pruning_ratio))]
                attn_mask = (attn_metric > threshold)
                mlp_mask = (mlp_metric > threshold)
        else:
            attn_mask = torch.stack(attn_mask) 
            mlp_mask = torch.stack(mlp_mask)
        
        mlp_raito = mlp_mask.view(-1).sum()/mlp_mask.numel()
        attn_ratio = attn_mask.view(-1).sum()/attn_mask.numel()
        print('remaining ratio:', pruning_ratio, 'mlp non_zero ratio:', mlp_raito, 'attn non_zero ratio:', attn_ratio)

        for name in mask.keys():
            if 'mlp' in name:
                if 'mlp.up_proj' in name or 'mlp.gate_proj' in name:
                    layer_id =int(re.search(r'layers\.(\d+)', name)[0].removeprefix('layers.'))
                    
                    mask[name][pruning_ratio] = mlp_mask[layer_id].cpu().numpy()
                    bias[name] = None
                    
                elif 'mlp.down_proj' in name:
                    layer_id = int(re.search(r'layers\.(\d+)', name)[0].removeprefix('layers.'))
                    
                    mask[name] = None
                    
                    output_weight = layers[layer_id].mlp.down_proj.weight
                    output_weight = dequantize_bnb_weight(output_weight, state=output_weight.quant_state)
                    
                    output_bias = ((mlp_baseline_inp_list[layer_id] * ~mlp_mask[layer_id].to(device)).to(output_weight) @ output_weight.T)
                    if args.wanda_sp:
                        output_bias = torch.zeros_like(output_bias)
                    bias[name][pruning_ratio] = output_bias.cpu().numpy()
                else:
                    print(f'No such param: {name}, please check the model')
                    sys.exit()
                    
            elif 'self_attn' in name:
                if 'self_attn.q_proj' in name or 'self_attn.k_proj' in name or 'self_attn.v_proj' in name:
                    layer_id = int(re.search(r'layers\.(\d+)', name)[0].removeprefix('layers.'))
                    
                    mask[name][pruning_ratio] = attn_mask[layer_id].repeat_interleave(128).cpu().numpy()
                    bias[name] = None
                    
                elif 'self_attn.o_proj' in name:
                    layer_id = int(re.search(r'layers\.(\d+)', name)[0].removeprefix('layers.'))
                    
                    mask[name] = None
                    
                    output_weight = layers[layer_id].self_attn.o_proj.weight
                    output_weight = dequantize_bnb_weight(output_weight, state=output_weight.quant_state)
                    
                    output_bias = ((attn_baseline_inp_list[layer_id] * ~attn_mask[layer_id].repeat_interleave(128).to(device)).to(output_weight) @ output_weight.T)
                    if args.wanda_sp:
                        output_bias = torch.zeros_like(output_bias)
                    bias[name][pruning_ratio] = output_bias.cpu().numpy()
                else:
                    print(f'No such param: {name}, please check the model')
                    sys.exit()
            else:
                print(f'No such param: {name}, please check the model')
                sys.exit()
          
    # for idx in range(len(layers)):
    #     if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
    #         compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
    #     else:
    #         compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, unstr=args.unstr)
                
    #     if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}): 
    #         compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
    #     else:
    #         compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr=args.unstr)
    

    # for name in mask.keys():
    #     for pruning_ratio in [1, 7/8, 3/4, 5/8, 1/2]:  ## meaning the remaining ratio
    #         print(name, pruning_ratio)
    #         if mask[name] is not None:
    #             print('mask.shape:', mask[name][pruning_ratio].shape)
    #             print('mask non-zero ratio:', mask[name][pruning_ratio].sum()/mask[name][pruning_ratio].shape[0])
    #         else:
    #             print('mask: None')
                
    #         if bias[name] is not None:
    #             print('bias.shape:', bias[name][pruning_ratio].shape)
    #             print('bias non-zero ratio:', (bias[name][pruning_ratio]!=0).sum()/bias[name][pruning_ratio].shape[0])
    #         else:
    #             print('bias: None')
                
    #         input()
                
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    
    return mask, bias