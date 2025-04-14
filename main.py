from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse
import time
import random
import re

import sys
sys.path.insert(0, "./transformers/src")
sys.path.insert(0, "./peft/src")

# from huggingface_hub import login
# login()

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    StoppingCriteria
)

from datasets import load_dataset, Dataset, DatasetDict
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from eval_func import eval_mmlu, eval_mmlu_wrapper, eval_wikitext2_wrapper, eval_general_ppl_wrapper, eval_lm_eval_wrapper


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True
    

if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

cls_tasks = {
    'classification': ['sst2', 'sst5', 'MR', 'SUBJ', 'AGNews', 'TREC', 'CB', 'BoolQ'], # , 'DBPedia'],
    'multiple choice': ['hellaswag', 'ARCE', 'PIQA', 'ARCC', 'OB', 'COPA', 'CQA'],
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )

@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=34, metadata={"help": "Size of validation dataset."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=256,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    
    do_eval_wikitext2: bool = field(
        default=False, metadata={"help": "evaluate the ppl on wikitext2."}
    )

    do_lm_eval: Optional[bool]=field(
        default=False, 
        metadata={"help":"Evalute on lm-eval-harness."}
    )
    
    do_lm_eval_task : str = field(
        default="arc_easy,piqa,sciq", metadata={"help": "Evaluation tasks in lm-eval-harness."}
    )
    
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=2, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=False, metadata={"help": 'To train or not to train, that is the question?'})
    no_eval_orig: bool = field(default=False, metadata={"help": 'do not eval the original test dataset corresponding to the training dataset'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=2, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

    few_shot_number: int = field(default=0, metadata={"help": 'few shot numbers for classification tasks'})

    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": 'enable ddp_find_unused_parameters in Accelerator.'})

    enable_shrinking: bool = field(default=False, metadata={"help": 'Enable shrinkable LLM.'})
    
    shrinkable_width: bool = field(default=False, metadata={"help": 'Enable shrinkable width in addition to layers.'})
    width_choice: str = field(default='[1,7/8,3/4,5/8,1/2]', metadata={"help": 'the available width choices for shrinkable width.'})
    nonuniform_width: bool = field(default=False, metadata={"help": 'Training with nonuniform width across layers.'})
    first_width: bool = field(default=False, metadata={"help": 'An ablation study: Only active the first widths (channels) in each layer.'})

    min_num_layer: int = field(default=16, metadata={"help": 'The minimal number of layers.'})
    random_sample_num_layer: int = field(default=2, metadata={"help": 'The number of randomely sampled layers in each iteration.'})
    kd_weight: float = field(default=1, metadata={"help": 'weight coefficient of the KD loss.'})

    sample_per_dataset: int = field(default=2000, metadata={"help": 'samples per dataset when training on cls_combo and mc_combo.'})

    num_remain_layers: int = field(default=1, metadata={"help": 'number of final layers remained during layer skipping.'})

    distill_all_tokens: bool = field(default=False, metadata={"help": 'Distill both target and context tokens to small models.'})
    
    layer_pruning: bool = field(default=None, metadata={"help": 'whether to apply layer pruning.'})
        
    distill_method: str = field(default='sp', metadata={"help": 'distillation method: sp, gkd, atkd.'})

    unc_thres: float = field(default=0.5, metadata={"help": 'the threshold for the uncertainty coefficient in ATKD.'})

    layer_calib_dp: bool = field(default=False, metadata={"help": 'enable the calibration based on dynamic programming to get layer ranking.'})

    dp_keep_last_layer: int = field(default=-1, metadata={"help": 'the last n layers to remain during dynamic programming.'})

    calib_dataset: str = field(default='wikitext2', metadata={"help": 'the dataset used for calibration.'})

    calib_metric: str = field(default=None, metadata={"help": 'the metric for calibration.'})

    width_calib: bool = field(default=False, metadata={"help": 'enable the calibration to get width ranking.'})

    prune_width_dim: str = field(default='in', metadata={"help": 'the width pruning dimension: {in, out}.'})

    prune_width_method: str = field(default='flap', metadata={"help": 'width pruning method: {wanda, flap}.'})

    wanda_sp: bool = field(default=False, metadata={"help": 'An ablation study to use wand-sp for pruning.'})

    num_calib_sample: int = field(default=20, metadata={"help": 'number of samples used for calibration.'})

    shrinking_method: str = field(default='first_layers', metadata={"help": 'the way to perform layer shrinking: {first_layers, calib, calib_dp}.'})
    
    shrinking_file: str = field(default=None, metadata={"help": 'the path to the file specifying the shrinking configuration.'})

    use_moe_lora: bool = field(default=False, metadata={"help": 'Use mixture of LoRA.'})
    
    moe_num_expert: int = field(default=5, metadata={"help": 'number of experts in MoE.'})

    moe_topk: int = field(default=2, metadata={"help": 'topk in MoE.'})

    resume_training: bool = field(default=False, metadata={"help": 'resume training from the latest checkpoint.'})
    
    distill_steps: int = field(default=-1, metadata={"help": 'number of training steps that enable distillation.'})

    no_balancing: bool = field(default=False, metadata={"help": 'ablation study: do not use loss balancing.'})
    
    eval_num_layer: int = field(default=24, metadata={"help": 'number of layers for evaluation.'})

    eval_num_width: float = field(default=0.875, metadata={"help": 'width for evaluation.'})


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

# hkolisetty6: NOT NEEDED for inference
def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

# hkolisetty6: NOT NEEDED for inference
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

# hkolisetty6: NEWLY ADDED
def get_accelerate_model_for_inference(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    shrink_config = {'enable_shrinking': args.enable_shrinking, 
                     "shrinkable_width": args.shrinkable_width,
                     "shrinking_method": args.shrinking_method,
                     "shrinking_file": args.shrinking_file,
                     "mask_dtype": "torch.float16" if args.fp16 else ("torch.bfloat16" if args.bf16 else "torch.float32")}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if not args.full_finetune else None,
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
        shrink_config = shrink_config
    )



def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}
    
    # device_map={'':torch.cuda.current_device()}

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    
    shrink_config = {'enable_shrinking': args.enable_shrinking, 
                     "shrinkable_width": args.shrinkable_width,
                     "shrinking_method": args.shrinking_method,
                     "shrinking_file": args.shrinking_file,
                     "mask_dtype": "torch.float16" if args.fp16 else ("torch.bfloat16" if args.bf16 else "torch.float32")}
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if not args.full_finetune else None,
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
        shrink_config = shrink_config
    )
        
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    if "decapoda-research-llama-7B-hf" in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=True, # False, # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="right",
            use_fast=True, # False, # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )
    
        
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                # "unk_token": tokenizer.convert_ids_to_tokens(
                #     model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
                # ),
        })
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)

        else:
            print(f'adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                
                use_moe_lora=args.use_moe_lora,
                num_experts=args.moe_num_expert,
                top_k=args.moe_topk,
                width_choice=args.width_choice if args.shrinkable_width else None,
            )
            model = get_peft_model(model, config)
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
            padding='max_length', # hkolisetty6: pad to max length to ensure constant length across batches
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        
        input_ids = []
        labels = []

        source_ids = []

        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:  ## by default: do not train on source
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )

                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
            
            source_ids.append(torch.tensor(tokenized_source))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        source_ids = pad_sequence(source_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # source_labels = pad_sequence(source_labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        data_dict = {
            'input_ids': input_ids,
            # 'source_ids': source_ids, # Not required for inference
            # 'source_labels': source_labels,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        if dataset_name == 'alpaca-gpt4':
            return load_dataset("vicgalle/alpaca-gpt4")
        if dataset_name == 'c4':
            return load_dataset("c4", 'en')
        if dataset_name == 'redpajama':
            return load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")

        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or dataset_format == 'alpaca-gpt4' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean', 'alpaca-gpt4'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'c4' or (dataset_format is None and args.dataset == 'c4'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'redpajama' or (dataset_format is None and args.dataset == 'redpajama'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        
        elif args.dataset in ['cls_combo', 'mc_combo']:
            dataset = dataset.map(lambda x: {
                'input': x['text'],
                'output': x['label'].strip(),
            })
            dataset = DatasetDict({"train": dataset})
            
        elif dataset_format == 'input-output':
            # leave as is
            pass
        
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
        )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)
    
    # for i in range(3):
    #     print('Input:')
    #     print(dataset['train'][i]['input'])
    #     print('Output:')
    #     print(dataset['train'][i]['output'])
    #     input()

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
            
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        # if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

class TokenTimingStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.prev_token_pred_time = time.perf_counter()
        self.ttft = {} # batch_num -> (batch_size, time)
        self.tbt = {} # batch_num -> (batch_size, num_tokens, average_time)
        self.num_tokens = 0
        self.batch_num = 0
        self.prev_batch_size = None
        self.total_time = 0
        self.prev_input_ids = None
        self.window_size = 5

    def __call__(self, input_ids, scores, **kwargs):
        batch_size, _ = input_ids.shape

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        current_time = time.perf_counter()
        
        if self.prev_input_ids is None or \
            self.prev_input_ids.shape[0] != batch_size or \
                ~torch.all(self.prev_input_ids[:, -self.window_size:] == input_ids[:, -self.window_size - 1 : -1]):
            
            # print input_ids shape
            print(f"input_ids shape: {input_ids.shape}")

            if self.prev_batch_size is not None:
                avg_time = self.total_time / self.num_tokens
                self.tbt[self.batch_num] = (self.prev_batch_size, self.num_tokens, avg_time)
                self.total_time = 0

            self.batch_num += 1
            self.ttft[self.batch_num] = (batch_size, current_time - self.prev_token_pred_time)
            self.num_tokens = 0
            self.prev_batch_size = batch_size
        else:
            self.num_tokens += 1
            self.total_time += current_time - self.prev_token_pred_time

        self.prev_input_ids = input_ids
        self.prev_token_pred_time = current_time
        return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
      
def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    
    print(args)
    
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    if args.width_calib:
        from width_shrink.prune import prune_wanda, prune_flap

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        if args.prune_width_method == 'wanda':
            mask = prune_wanda(args, model, tokenizer)
            np.save(f'{args.output_dir}/width_mask.npy', mask)
        elif args.prune_width_method == 'flap':
            mask, bias = prune_flap(args, model, tokenizer)
            np.save(f'{args.output_dir}/width_mask.npy', mask)
            np.save(f'{args.output_dir}/width_bias.npy', bias)
        else:
            print('Wrong pruning method:', args.prune_width_method)
            sys.exit()

        print('Finished width calibration.')
        
        return 

    if args.shrinkable_width:
        print('Setting width mask and bias...')
        
        shrink_file = np.load(args.shrinking_file, allow_pickle=True).item()
        assert 'width_mask' in shrink_file
        width_mask = shrink_file['width_mask']
        
        if args.prune_width_method == 'flap':
            bias = shrink_file['bias']
            
        if args.first_width:
            for key, mask_dict in width_mask.items():
                for ratio, mask in mask_dict.items():
                    width_mask[key][ratio] = np.sort(mask)[::-1]
                
        for name, module in model.named_modules():
            if name in width_mask:
                mask_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
                if args.prune_width_method == 'flap':
                    if 'mlp.down_proj' in name or 'self_attn.o_proj' in name:
                        assert width_mask[name] is None
                        for key in bias[name].keys():
                            # assert key in eval(args.width_choice)
                            bias[name][key] = torch.from_numpy(bias[name][key]).to(mask_dtype)
                            
                        module.set_width_mask(width_mask=None, output_bias=bias[name])
                    else:
                        assert bias[name] is None
                        for key in width_mask[name].keys():
                            # assert key in eval(args.width_choice)
                            width_mask[name][key] = torch.from_numpy(width_mask[name][key]).to(mask_dtype)
                            
                        module.set_width_mask(width_mask=width_mask[name], output_bias=None)
                
                elif args.prune_width_method == 'wanda':
                    for key in width_mask[name].keys():
                        # assert key in eval(args.width_choice)
                        width_mask[name][key] = torch.from_numpy(width_mask[name][key]).to(mask_dtype)
                    module.set_width_mask(width_mask=width_mask[name])

                else:
                    print('No such pruning method:', args.prune_width_method)
                    sys.exit()
                    
    data_module = make_data_module(tokenizer=tokenizer, args=args)
    
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
        
    if args.do_mmlu_eval or args.calib_dataset == 'mmlu':
        if args.mmlu_dataset == 'mmlu-zs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/zero_shot_mmlu_val.json',
                'test': 'data/mmlu/zero_shot_mmlu_test.json',
            })
            mmlu_dataset = mmlu_dataset.remove_columns('subject')
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
            mmlu_dataset = load_dataset("json", data_files={
                'eval': 'data/mmlu/five_shot_mmlu_val.json',
                'test': 'data/mmlu/five_shot_mmlu_test.json',
            })
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")
        if args.do_train:
            class MMLUEvalCallback(transformers.TrainerCallback):
                def on_evaluate(self, args, state, control, model, **kwargs):
                    source_max_len = trainer.data_collator.source_max_len
                    trainer.data_collator.source_max_len = args.mmlu_source_max_len

                    if args.enable_shrinking:
                        active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, 0)

                        if args.full_finetune:
                            model.set_active_layers(active_layers_attn_list[0], active_layers_mlp_list[0])
                        else:
                            model.set_active_layers(active_layers_attn_list[0], active_layers_mlp_list[0], width=1)
                    
                        results_largest = eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy)
                        results_largest = {k + "_largest": v for k, v in results_largest.items()}

                        results = results_largest

                    else:
                        results = eval_mmlu(trainer, mmlu_dataset, args, abcd_idx, accuracy)

                    trainer.log(results)
                    trainer.mmlu_results = results
                    trainer.data_collator.source_max_len = source_max_len

            trainer.add_callback(MMLUEvalCallback)
    else:
        mmlu_dataset = None
        abcd_idx = None
        accuracy = None

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)
        
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train(resume_from_checkpoint=checkpoint_dir if args.resume_training else None)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

                
    if (args.do_eval and not args.no_eval_orig) or args.do_predict:
        if args.enable_shrinking:
            active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, 0)

            model.set_active_layers(active_layers_attn_list[0], active_layers_mlp_list[0], width=1)
            
            if args.shrinkable_width:
                for module in model.modules():
                    if hasattr(module, 'set_width_ratio'):
                        module.set_width_ratio(width_ratio=eval(args.width_choice)[0])
        if args.do_eval:                        
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            all_metrics.update(metrics)
        

    if args.layer_calib_dp:
        assert args.enable_shrinking
        
        if args.calib_dataset in ['wikitext2', 'redpajama', 'bookcorpus']:
            metric = 'loss'
        elif args.calib_dataset in ['mmlu']:
            if args.calib_metric == 'loss':
                metric = 'loss'
            else:
                metric = 'acc'

        elif args.calib_dataset == 'redpajama':
            calib_dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
        elif args.calib_dataset == 'bookcorpus':
            calib_dataset = load_dataset("bookcorpus")
        else:
            raise ValueError(f"Unknown dataset: {args.calib_dataset}")
        
        
        if args.dp_keep_last_layer > 0:
            N = model.config.num_hidden_layers - args.dp_keep_last_layer  # total number of layers
            M = model.config.num_hidden_layers - args.min_num_layer  # Maximum number of layers to remove
            offset = np.ones(args.dp_keep_last_layer)
        else:
            N = model.config.num_hidden_layers  # total number of layers
            M = N - args.min_num_layer  # Maximum number of layers to remove
            offset = None
                    
        if metric == 'loss':
            d = np.full((N+1, M+1), float('inf'))
        else:
            d = np.full((N+1, M+1), float('-inf'))
            
        strategy = np.zeros((N+1, M+1), dtype=object)

        # Boundary condition: Perplexity of removing 0 layers is set to -1
        for i in range(N+1):
            strategy[i][0] = np.ones(N)  # All layers active up to that point
        
        # Fill the dynamic programming table
        for n in range(1, N+1):
            for m in range(1, M+1):
                # Only try to remove a layer if it's possible to have removed m layers before n
                if m <= n:
                    new_active_layers = strategy[n-1][m-1].copy()
                    new_active_layers[N-n] = 0
                    
                    if offset is not None:
                        active_layers_attn = active_layers_mlp = np.concatenate((new_active_layers, offset), axis=0)
                    else:
                        active_layers_attn = active_layers_mlp = new_active_layers
                        
                    model.set_active_layers(active_layers_attn, active_layers_mlp, width=1)
                    
                    model.eval()
                    with torch.no_grad(): 
                        if args.calib_dataset == 'wikitext2':
                            results = eval_wikitext2_wrapper(trainer, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['wikitext2_ppl']
                        elif args.calib_dataset == 'redpajama':
                            results = eval_general_ppl_wrapper(trainer, calib_dataset, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['ppl']
                        elif args.calib_dataset == 'bookcorpus':
                            results = eval_general_ppl_wrapper(trainer, calib_dataset, tokenizer, model, n_samples=args.num_calib_sample)
                            current_metric = results['ppl']
                        elif args.calib_dataset == 'mmlu':
                            results = eval_mmlu_wrapper(trainer, mmlu_dataset, args, abcd_idx, accuracy, n_samples=args.num_calib_sample if args.max_mmlu_samples is None else None)
                            
                            if metric == 'loss':
                                current_metric = results['mmlu_loss']
                            else:
                                current_metric = results['mmlu_eval_accuracy']
                        else:
                            print("Not implemented calibration dataset:", args.calib_dataset)
                            sys.exit()

                    if metric == 'loss': # the lower the better
                        if current_metric < d[n-1][m]:
                            d[n][m] = current_metric
                            strategy[n][m] = new_active_layers
                        else:
                            d[n][m] = d[n-1][m]
                            strategy[n][m] = strategy[n-1][m].copy()
                    else:  # the higher the better
                        if current_metric > d[n-1][m]:
                            d[n][m] = current_metric
                            strategy[n][m] = new_active_layers
                        else:
                            d[n][m] = d[n-1][m]
                            strategy[n][m] = strategy[n-1][m].copy()

        if offset is not None:
            final_strategy = {'strategy': {m: np.concatenate((strategy[N][m], offset), axis=0) for m in range(1, M+1)}, 'metric': {m: d[N][m] for m in range(1, M+1)}}
        else:
            final_strategy = {'strategy': {m: strategy[N][m] for m in range(1, M+1)}, 'metric': {m: d[N][m] for m in range(1, M+1)}}

        print('final_strategy:', final_strategy)
        np.save(os.path.join(args.output_dir, 'final_strategy.npy'), final_strategy)
        
        np.save(os.path.join(args.output_dir, 'full_strategy.npy'), strategy)
        np.save(os.path.join(args.output_dir, 'metric.npy'), d)
    

    if args.enable_shrinking:
        ####################### Manually set up the layer and width choices here #########################
        eval_num_layer = [args.eval_num_layer]  # [32, 30, 28, 26, 24, 22, 20, 18, 16]
        width_list = [args.eval_num_width] # [1, 7/8, 3/4, 5/8, 1/2]
        #########################################################################################
        
        if eval_num_layer is None:
            if model.config.num_hidden_layers == args.min_num_layer:
                eval_num_layer = [model.config.num_hidden_layers]
            elif args.random_sample_num_layer > 0:
                if args.min_num_layer >= 20:
                    eval_num_layer = [model.config.num_hidden_layers, 24, args.min_num_layer]
                else:
                    eval_num_layer = [model.config.num_hidden_layers, 24, 20, args.min_num_layer]
            else:
                eval_num_layer = [model.config.num_hidden_layers, args.min_num_layer]
            

        if args.shrinking_method == 'calib_dp':
            strategy = np.load(args.shrinking_file, allow_pickle=True).item()["strategy"]
            if 0 not in list(strategy.keys()):
                strategy[0] = np.ones(model.config.num_hidden_layers)

            active_layers_list = []
            for num_layer in eval_num_layer:
                active_layers_list.append(strategy[model.config.num_hidden_layers - num_layer])   # the key of self.strategy is the num of removed layers
            
            active_layers_attn_list = active_layers_mlp_list = active_layers_list

        elif args.shrinking_method == 'first_layers':
            active_layers_attn_list = active_layers_mlp_list = trainer.sandwich_sampling(model.config.num_hidden_layers, args.min_num_layer, random_sample_num=0, inter_choices=eval_num_layer[1:-1])
        
        else:
            print('Not implemented:', args.shrinking_method)
            sys.exit()


        if not args.shrinkable_width:
            for num_layer, active_layers_attn, active_layers_mlp in zip(eval_num_layer, active_layers_attn_list, active_layers_mlp_list):
                model.set_active_layers(active_layers_attn, active_layers_mlp)
                all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics, suffix=f'_l{num_layer}')
        
        else:
            if width_list is None:
                width_choice = eval(args.width_choice)
                
                if len(width_choice) == 1 or model.config.num_hidden_layers == args.min_num_layer: # if the num layer is not shrinkable, we will measure all candidate widths
                    width_list = width_choice
                    
                else:
                    width_list = [1, 3/4, width_choice[-1]]
            
            for num_layer, active_layers_attn, active_layers_mlp in zip(eval_num_layer, active_layers_attn_list, active_layers_mlp_list):   
                for width in width_list:
                    for module in model.modules():
                        if hasattr(module, 'set_width_ratio'):
                            module.set_width_ratio(width_ratio=width)
                    
                    model.set_active_layers(active_layers_attn, active_layers_mlp, width=width)

                    if args.do_eval:
                        all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics, suffix=f'_l{num_layer}w{width}')
                        
    else:
        all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics)
                    
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        timing_stopping_criteria = TokenTimingStoppingCriteria()
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict", stopping_criteria=[timing_stopping_criteria])
        
        print(timing_stopping_criteria.ttft)
        print(timing_stopping_criteria.tbt)

        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        # predictions = prediction_output.label_ids
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        
        # predictions = [ 
        #     ''.join(tokenizer.batch_decode(prediction.astype(int), skip_special_tokens=True, clean_up_tokenization_spaces=True))
        #     for prediction in predictions
        # ]

        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_mmlu_eval or args.do_predict or args.do_lm_eval):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


def eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx, accuracy, all_metrics, suffix=''):
    model.eval()
    
    with torch.no_grad():
        if args.do_lm_eval:
            results = eval_lm_eval_wrapper(trainer, tokenizer, model, args, suffix=suffix)
            all_metrics.update(results)
                    
        if args.do_mmlu_eval:   
            results = eval_mmlu_wrapper(trainer, mmlu_dataset, args, abcd_idx, accuracy, suffix=suffix)
            all_metrics.update(results)

        if args.do_eval_wikitext2:
            results = eval_wikitext2_wrapper(trainer, tokenizer, model, suffix=suffix)
            all_metrics.update(results)
        
    return all_metrics


    
if __name__ == "__main__":
    train()

