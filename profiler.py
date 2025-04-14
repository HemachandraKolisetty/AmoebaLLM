import os

from os.path import isdir, join
import numpy as np
import json
import logging

import torch
import transformers
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    set_seed,
)
import evaluate

from peft import (
    PeftModel
)
from peft.tuners.lora import LoraLayer

from main import (
    ModelArguments, 
    DataArguments, 
    TrainingArguments, 
    GenerationArguments, 
    smart_tokenizer_and_embedding_resize,
    DEFAULT_PAD_TOKEN,
    make_data_module,
    load_dataset,
    TokenTimingStoppingCriteria,
    eval_all,
)

def get_compute_dtype(args):
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if torch.cuda.is_bf16_supported() and compute_dtype == torch.float16:
        print('GPU supports bfloat16, so switching the compute_dtype to torch.bfloat16')
        compute_dtype = torch.bfloat16
    return compute_dtype

def parse_args(args_list=None):
    '''
    Parse the command line arguments for the Hugging Face model, data, training, and generation arguments
    '''
    # Create an argument parser for the Hugging Face model, data, training, and generation arguments
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))

    # Parse the command line arguments into the respective dataclass instances
    if args_list is None:
        model_args, data_args, training_args, generation_args, _ = \
            hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    else:
        model_args, data_args, training_args, generation_args, _ = \
            hfparser.parse_args_into_dataclasses(args_list, return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )

    # Combine the parsed arguments into a single argparse.Namespace object
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # Set additional parameters for later use
    args.compute_dtype = get_compute_dtype(args)

    return args, training_args

def get_last_checkpoint(output_dir):
    '''
    Given the output directory, return the last checkpoint directory
    '''
    assert output_dir is not None, 'Output directory must be specified'
    assert isdir(output_dir), 'Output directory does not exist'

    max_steps = 0
    for filename in os.listdir(output_dir):
        if isdir(join(output_dir, filename)) and filename.startswith('checkpoint-'):
            step = int(filename.split('-')[-1])
            if step > max_steps:
                max_steps = step
    assert max_steps > 0, 'No checkpoints found in output directory'
    last_checkpoint_dir = join(output_dir, f'checkpoint-{max_steps}')
    return last_checkpoint_dir

def get_tokenizer(args, model):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        token=args.use_auth_token,
        padding_side='right',
        tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
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
        print('Adding correct special tokens to the llama tokenizer')
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
        })
    return tokenizer

# Assumes the pruning width method is flap
def set_width_mask_and_bias(model, args):
    '''
    Set the width_mask and bias for the model

    hkolisetty6:
    My understanding is that each module will hold the width_mask and bias for the corresponding layer.
    The width_mask and bias are dicts with keys as width_ratio and values as the mask/bias tensor.
    During inference, the width_mask and bias are applied to the output of the module.
    In essense, full matrix multiplication is done and then the width_mask and bias are applied 
    to the output before passing it to the next layer.
    '''
    shrink_file = np.load(args.shrinking_file, allow_pickle=True).item()
    assert 'width_mask' in shrink_file, 'Width mask not found in shrinking file'
    assert 'bias' in shrink_file, 'Bias not found in shrinking file'

    width_mask = shrink_file['width_mask']
    bias = shrink_file['bias']

    for name, module in model.named_modules():
        if name in width_mask:
            mask_dtype = args.compute_dtype # TODO hkolisetty6: in original code, this is set to torch.float32 when args.fp16 is True
            if 'mlp.down_proj' in name or 'self_attn.o_proj' in name:
                assert width_mask[name] is None
                for key in bias[name].keys():
                    bias[name][key] = torch.from_numpy(bias[name][key]).to(mask_dtype)
                module.set_width_mask(width_mask=None, output_bias=bias[name])
            else:
                assert bias[name] is None
                for key in width_mask[name].keys():
                    width_mask[name][key] = torch.from_numpy(width_mask[name][key]).to(mask_dtype)
                module.set_width_mask(width_mask=width_mask[name], output_bias=None)


# Assumes model is loaded on a single GPU
def load_model(args, checkpoint_dir):
    '''
    Given the command line arguments and the checkpoint directory, return the model
    '''
    shrink_config = {
        'enable_shrinking': args.enable_shrinking,
        'shrinkable_width': args.shrinkable_width,
        'shrinking_method': args.shrinking_method,
        'shrinking_file': args.shrinking_file,
        'mask_dtype': str(args.compute_dtype),
    }
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=args.compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quant_type,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        device_map='auto',
        torch_dtype=args.compute_dtype, # TODO hkolisetty6: in original code, this is set to torch.float32
        token=args.use_auth_token,
        shrink_config=shrink_config,
        quantization_config=quantization_config,
    )
    model.config.torch_dtype = args.compute_dtype

    # !IMPORTANT 
    # Load the tokenizer before loading the adapters to ensure that the special tokens, if not available, are added to embeddings
    # This is important for the adapters to be loaded correctly
    tokenizer = get_tokenizer(args, model)

    # Load adapters from checkpoint
    print('Loading adapters from checkpoint')
    model = PeftModel.from_pretrained(
        model=model,
        model_id=join(checkpoint_dir, 'adapter_model'),
    )

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) or 'lm_head' in name or 'embed_tokens' in name:
            if args.compute_dtype == torch.bfloat16:
                module = module.to(torch.float32)
        if 'norm' in name:
            module = module.to(torch.float32)
    
    model.config.use_cache = False # Check LlamaConfig for this attribute

    # Set the width mask and bias for the model
    set_width_mask_and_bias(model, args)

    # deebert_dynamo_model = torch.compile(model, backend="inductor")
    # saved_deebert_dynamo = "deebert_dynamo_min_batch_mnli_dataset.pt"
    # torch.save(deebert_dynamo_model, saved_deebert_dynamo)

    # TODO: remove these lines
    # compiled_model = torch.compile(model, backend='inductor')
    # torch.save(compiled_model, 'compiled_model.pt')

    # scripted_model = torch.jit.load('compiled_model.pt')

    return model, tokenizer

def get_mmlu_dataset_for_evaluation(args, tokenizer):
    if args.do_mmlu_eval:
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
        return mmlu_dataset, abcd_idx, accuracy
    return None, None, None

def eval_model(trainer, args, logger, all_metrics):
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

# Assumes shrinking_method is calib_dp and shrinking is enabled
def setup_model_for_inference(model, args):
    '''
    Setup model for inference by activating the layers and setting the width ratio
    '''
    strategy = np.load(args.shrinking_file, allow_pickle=True).item()["strategy"]
    if 0 not in list(strategy.keys()):
        strategy[0] = np.ones(model.config.num_hidden_layers)

    active_layers_attn = active_layers_mlp = strategy[
        model.config.num_hidden_layers - args.eval_num_layer
    ]

    if args.shrinkable_width:
        for module in model.modules():
            if hasattr(module, 'set_width_ratio'):
                module.set_width_ratio(width_ratio=args.eval_num_width)
        model.set_active_layers(
            active_layers_attn, active_layers_mlp, width=args.eval_num_width
        )
    else:
        model.set_active_layers(active_layers_attn, active_layers_mlp)


def profile_latencies(model, tokenizer, args, logger, trainer, data_module):
    '''
    Given depth and width ratios (in args), profile the model for TTFT and TBT latencies
    '''
    setup_model_for_inference(model, args)
    logger.info("Profiling model for TTFT and TBT latencies")
    timing_stopping_criteria = TokenTimingStoppingCriteria()
    prediction_output = trainer.predict(
        test_dataset=data_module["predict_dataset"],
        metric_key_prefix="predict",
        stopping_criteria=[timing_stopping_criteria],
    )

    print(timing_stopping_criteria.ttft)
    print(timing_stopping_criteria.tbt)

    prediction_metrics = prediction_output.metrics
    predictions = prediction_output.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

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

    return timing_stopping_criteria.ttft, timing_stopping_criteria.tbt

def profile_accuracies(model, tokenizer, args, trainer, logger, mmlu_dataset, abcd_idx, accuracy, all_metrics):
    logger.info("Profiling model for accuracies")
    setup_model_for_inference(model, args)
    num_layer = args.eval_num_layer
    width = args.eval_num_width
    all_metrics = eval_all(args, model, trainer, tokenizer, mmlu_dataset, abcd_idx=abcd_idx, accuracy=accuracy, all_metrics=all_metrics, suffix=f'_l{num_layer}w{width}')

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))
    return all_metrics
    
def get_latency_stats(ttft, tbt, bs):
    '''
    ttft (time-to-first-token) is a dictionary with keys as batch_num and values as tuples (batch_size, latency)
    tbt (time-between-tokens) is a dictionary with keys as batch_num and values as tuples (batch_size, num_tokens, avg_latency)

    Returns:
    - batch_size
    - mean_ttft
    - std_ttft
    - mean_tbt
    - std_tbt

    Excludes the first 5 batches from both ttft and tbt
    Excluded the last batch from ttft (since it is not a full batch)
    - No need to exclude the last batch from tbt since the last batch is not included in tbt
    '''
    ttft_latencies = []
    tbt_latencies = []
    for (_, latency) in ttft.values():
        ttft_latencies.append(latency * 1e6) # Convert to microseconds
    
    for (_, _, avg_latency) in tbt.values():
        tbt_latencies.append(avg_latency * 1e6) # Convert to microseconds

    ttft_latencies = ttft_latencies[20:-1]
    tbt_latencies = tbt_latencies[20:]

    return {
        'batch_size': bs,
        'mean_ttft': np.mean(ttft_latencies),
        'std_ttft': np.std(ttft_latencies),
        'mean_tbt': np.mean(tbt_latencies),
        'std_tbt': np.std(tbt_latencies),
    }
