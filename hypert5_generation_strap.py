import sys
import torch
import datasets
import json
import logging
import os
import argparse
import re, regex
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy

from hyperformer.third_party.models import T5Config, T5ForConditionalGeneration
from hyperformer.third_party.trainers import T5Trainer
from hyperformer.third_party.trainers import T5Generator

from hyperformer.adapters import AdapterController, AutoAdapterConfig
from hyperformer.data import AutoTask
from hyperformer.third_party.utils import TaskCollator, check_output_dir
from hyperformer.metrics import build_compute_metrics_fn
from hyperformer.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments
from hyperformer.utils import freezing_params, get_last_checkpoint_path, create_dir,\
    handle_metrics, get_training_args


def tweet_normalizer(txt):
    txt.replace("<unk>","").replace("<unk>","<pad>")
    # remove duplicates
    temp_text = regex.sub("(USER\s+)+","USER ", txt)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    temp_text = re.sub("[\r\n\f\t]+","",temp_text)
    temp_text = re.sub(r"\s+"," ", temp_text)
    temp_text = regex.sub("(USER\s+)+","USER ", temp_text)
    temp_text = regex.sub("(URL\s+)+","URL ", temp_text)
    
    return temp_text

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--config_file", default=None, type=str, required=True,
                    help="Path to config file")

    parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="Path to pre-trained model name")

    parser.add_argument("--source_cls", default=None, type=str, required=True,
                    help="Prefix of source input")

    parser.add_argument("--target_cls", default=None, type=str, required=True,
                    help="Prefix of target style")
    
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size GPU/CPU.")

    parser.add_argument("--num_workers", default=1, type=int,
                        help="Total number of num_workers.")

    parser.add_argument("--num_return", default=1, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_k", default=0, type=int,
                        help="Total number of generated paraphrases per tweet.")

    parser.add_argument("--top_p", default=0.95, type=float,
                        help="Total number of generated paraphrases per tweet.")


    args = parser.parse_args()
    
    config = json.load(open(args.config_file, "r"))
    del config['local_rank']
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_dict(config)

    config = T5Config.from_pretrained(
        training_args.output_dir, 
        local_files_only = True
    )
    
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout",  "train_adapters")

    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))
            
            
    # Gets the adapter config and updates the specified parameters.
    if training_args.train_adapters:
        adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
        adapter_config.input_dim = config.d_model
        adapter_config.tasks = data_args.tasks
        adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(data_args.tasks, data_args.adapters)} if data_args.adapters is not None else None
        # If this is a parametric task embedding this mapping makes sense, but in case we use any task embeddings,
        # then, we do not need any mapping as we use the pretrained task embeddings.
        adapter_config.task_to_embeddings = {task:embedding for task, embedding in zip(data_args.tasks, data_args.task_embeddings)}\
             if (data_args.task_embeddings is not None) else None
        extra_adapter_params = ("task_embedding_dim",
                                "add_layer_norm_before_adapter",
                                "add_layer_norm_after_adapter",
                                "reduction_factor",
                                "hidden_dim",
                                "non_linearity",
                                "train_task_embeddings",
                                "projected_task_embedding_dim",
                                "task_hidden_dim",
                                "conditional_layer_norm",
                                "train_adapters_blocks",
                                "unique_hyper_net",
                                "unique_hyper_net_layer_norm",
                                "efficient_unique_hyper_net")
        for p in extra_adapter_params:
            if hasattr(adapter_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(adapter_args, p))
            else:
                logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
        adapter_config.device = training_args.device
    else:
        adapter_config = None
        
    ### Load checkpint   
    last_checkpoint_path = training_args.output_dir

    model_path = model_args.model_name_or_path if ((training_args.optimize_from_scratch and not training_args.optimize_from_scratch_with_loading_model) or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
        else last_checkpoint_path

    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        cache_dir=model_args.cache_dir,
        adapter_config=adapter_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
        model_path,
        cache_dir=model_args.cache_dir
    )
    
    
    file_name = args.input_file.split("/")[-1].replace(".tsv", "")
    output_file = os.path.join(args.output_dir, "{}-{}_{}-{}_{}_transfer.json".format("test",args.model_name, args.source_cls, args.target_cls, str(args.top_p)))
    
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams
        
    dataset_class = AutoTask(model_args.cache_dir)

    test_dataset = {args.source_cls: dataset_class.get(args.source_cls, args.input_file, seed=data_args.data_seed).get_dataset(
                split="test",
                add_prefix=False,
                split_validation_test=False)} 

    
    generator = T5Generator(
        model=model,
        target_type = args.target_cls,
        top_p=args.top_p,
        config = config,
        args = training_args,
        eval_dataset=test_dataset,
        data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
        data_args=data_args,
        dataset_sizes=None,
        adapter_config=adapter_config
    )
    
    predictions, labels = generator.predict(test_dataset)
    
    if generator.is_world_process_zero():
        test_preds = tokenizer.batch_decode(
            predictions[args.source_cls], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        test_preds = [pred.strip() for pred in test_preds]

        test_labels = tokenizer.batch_decode(
            labels[args.source_cls], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
        test_labels = [pred.strip() for pred in test_preds]
        
        output_lines = []
        
        for ind in range(len(test_labels)):
            para_text = test_dataset[args.source_cls][ind]["src_texts"]
            org_tweet = test_dataset[args.source_cls][ind]["tgt_texts"]

            paraphrases = test_preds[ind * args.num_return : (ind+1) * args.num_return]
            paraphrases = [tweet_normalizer(x) for x in paraphrases]

            output_all = {}
            output_all["para_text"] = para_text
            output_all["org_tweet"] = org_tweet
            output_all["source_cls"] = args.source_cls.split("-")[-1] if "-" in args.source_cls else args.source_cls
            output_all["target_cls"] = args.target_cls.split("-")[-1] if "-" in args.target_cls else args.target_cls

            output_all["paraphrase"] = paraphrases[0]

            output_lines.append(json.dumps(output_all)+"\n")

        with open(output_file, "a") as out_f:
            out_f.writelines(output_lines)

    
if __name__ == "__main__":
    main()
