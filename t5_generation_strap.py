import torch
from transformers import T5ForConditionalGeneration,T5TokenizerFast
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import GPUtil
import re, regex
import json, sys, regex
import argparse
import logging
import glob
import os
from tqdm import tqdm, trange
import pandas as pd
import torch.nn as nn

global device, device_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
device_ids = GPUtil.getAvailable(limit = 4)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, source_cls, target_cls):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = self.data.source
        self.org_text = self.data.target

        self.labels = self.data.label
        self.max_len = max_len

        self.source_cls = source_cls
        self.target_cls = target_cls

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        org_tweet =  str(self.org_text[index])

        label = str(self.labels[index])
        
        input_st = comment_text #.replace(self.source_cls+"_CLS", self.target_cls+"_CLS") 
        input_st = input_st+"</s>"

        encoding = self.tokenizer.encode_plus(input_st,pad_to_max_length=True, 
                                         return_tensors="pt", max_length=self.max_len,
                                        truncation=True)

        ids = encoding['input_ids']
        mask = encoding['attention_mask']

        return {
            'para_text': comment_text,
            'org_tweet': org_tweet,
            'input_ids': ids,
            'attention_mask': mask,
            'label': label
        }

def regular_encode(args, file_path, tokenizer, shuffle=True, num_workers = 1, batch_size=64, maxlen = 32, mode = 'train'):
    
    # if we are in train mode, we will load two columns (i.e., text and label).
    if mode == 'train':
        # Use pandas to load dataset
        df = pd.read_csv(file_path, delimiter='\t',header=0, names=['source', 'target', 'label'], encoding='utf-8', quotechar='"')

    else:
        print("the type of mode should be either 'train' or 'predict'. ")
        return
        
    print("{} Dataset: {}".format(file_path, df.shape))
    
    custom_set = CustomDataset(df, tokenizer, maxlen, args.source_cls, args.target_cls)
    
    dataset_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_workers}

    batch_data_loader = DataLoader(custom_set, **dataset_params)
    
    return batch_data_loader


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

    parser.add_argument("--model_path", default=None, type=str, required=True,
                    help="Path to pre-trained model or shortcut name")

    parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="Path to pre-trained model name")

    parser.add_argument("--source_cls", default=None, type=str, required=True,
                    help="Prefix of source input")

    parser.add_argument("--target_cls", default=None, type=str, required=True,
                    help="Prefix of target style")
    
    ## Other parameters
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
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

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = 0 if device=="cpu" else torch.cuda.device_count()

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)  
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

    if torch.cuda.is_available():
        if n_gpu == 1:
            model = model.to(device)
        else:
            torch.backends.cudnn.benchmark = True
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model

    train_file = args.input_file

    train_dataloader = regular_encode(args, train_file, tokenizer, batch_size=args.batch_size, maxlen = args.max_seq_length)

    file_name = args.input_file.split("/")[-1].replace(".tsv", "")
    
    output_file = os.path.join(args.output_dir, "{}-{}_{}-{}_{}_transfer.json".format(file_name, args.model_name, args.source_cls, args.target_cls, str(args.top_p)))
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    model.eval()
    for _, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        input_ids, attention_masks = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        labels = batch["label"]
        para_texts = batch["para_text"]
        org_tweets = batch["org_tweet"]

        beam_outputs = model.generate(
                input_ids=input_ids.squeeze(1), attention_mask=attention_masks.squeeze(1),
                do_sample=True,
                max_length=args.max_seq_length,
                top_k=args.top_k,
                top_p=args.top_p,
                early_stopping=True,
                num_return_sequences=args.num_return
            )

        beam_outputs = beam_outputs.cpu()
        org_inputs = input_ids.squeeze(1).cpu().numpy().tolist()

        final_outputs = [tokenizer.decode(x, skip_special_tokens=True,clean_up_tokenization_spaces=True) for x in beam_outputs]

        output_lines = []
        for ind in range(len(labels)):
            para_text = para_texts[ind]
            org_tweet = org_tweets[ind]

            paraphrases = final_outputs[ind * args.num_return : (ind+1) * args.num_return]
            paraphrases = [tweet_normalizer(x) for x in paraphrases]
            
            output_all = {}
            output_all["para_text"] = para_text
            output_all["org_tweet"] = org_tweet
            output_all["source_cls"] = args.source_cls
            output_all["target_cls"] = args.target_cls
            
            output_all["label"] = labels[ind]
            
            output_all["paraphrase"] = paraphrases[0]
            
            output_lines.append(json.dumps(output_all)+"\n")
            
        with open(output_file, "a") as out_f:
            out_f.writelines(output_lines)
    

if __name__ == "__main__":
    main()
