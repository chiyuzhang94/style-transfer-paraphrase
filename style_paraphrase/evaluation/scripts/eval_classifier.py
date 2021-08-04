#!/usr/bin/env python
# coding: utf-8

# (1)load libraries 
import json, sys, regex, csv
import torch
import GPUtil
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer 
from tqdm import tqdm, trange
import pandas as pd
import argparse
import os

##----------------------------------------------------
from transformers import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(device)


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default=None, type=str)
parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--output_path', type=str, default=None)

args = parser.parse_args()

args.output_path = args.output_path + "/" + args.input_file.split("/")[-1].replace(".json", "/")
print(args.output_path)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, lab2ind):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.lab2ind = lab2ind

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        comment_text = str(self.data[index]["paraphrase"])
        
        label = str(self.data[index]["target_cls"])
        
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': label
        }


# define a function for data preparation
def regular_encode(input_file, tokenizer, lab2ind, shuffle=False, num_workers = 0, batch_size=64, maxlen = 64):
    
    # if we are in predict mode, we will load one column (i.e., text).

        
    with open(input_file, "r") as f:
        author_data = f.read().strip().split("\n")
    author_data = [json.loads(x) for x in author_data] 

    
    custom_set = CustomDataset(author_data, tokenizer, maxlen,lab2ind)
    
    dataset_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}

    batch_data_loader = DataLoader(custom_set, **dataset_params)
    
    return batch_data_loader


def inference(model, iterator, infer_out_file, ind2label):
    model.eval()
    all_pred=[]

    all_predictions = []
    all_labels = []

    softmax = nn.Softmax(dim=1)
    
    with open(infer_out_file, 'w') as target:
        writer = csv.writer(target, delimiter='\t', quoting=csv.QUOTE_NONE)
        columns = ["target_cls", "prediction", "prob"]
        for k in range(len(ind2label)):
            columns.append(ind2label[k])

        writer.writerow(columns)
        
        with torch.no_grad():
            for _, batch in enumerate(tqdm(iterator, desc="Iteration")):
            # Add batch to GPU
                input_ids = batch['ids'].to(device, dtype = torch.long)
                input_mask = batch['mask'].to(device, dtype = torch.long)
                labels = batch['target']
                
                outputs = model(input_ids, input_mask)
                logits = softmax(outputs["logits"])

                # delete used variables to free GPU memory
                del batch, input_ids, input_mask
                # identify the predicted class for each example in the batch
                probabilities, predicted = torch.max(logits.cpu().data, 1)
                # put all the true labels and predictions to two lists

                for i in range(len(probabilities)):
                    tmp = []
                    tmp.append(labels[i])
                    tmp.append(ind2label[predicted[i].item()])
                    tmp.append(probabilities[i].item())
                    probs_all = [x.item() for x in logits.cpu().data[i] ]
                    
                    tmp.extend(probs_all)
                    all_pred.append(tmp)   

                    all_predictions.append(ind2label[predicted[i].item()])
                    all_labels.append(labels[i])

                    if len(all_pred) == 10000:
                        writer.writerows(all_pred)
                        all_pred = []      

        if len(all_pred) > 0:
            writer.writerows(all_pred)

    ncorrect = sum([1 if l1.lower() == l2.lower() else 0 for l1, l2 in zip(all_predictions, all_labels)])

    nsamples = len(all_labels)

    overall_accuracy = float(ncorrect) * 100 / float(nsamples)

    with open(infer_out_file.replace("tsv", "summary") , "w") as f:
        f.write("overall_accuracy: " + str(overall_accuracy) + "\n")

def main_run(args):
    #---------------------------------------

    modelbased = "roberta"
    max_seq_length= 60
    model_path=str(args.model_dir)
    input_file = str(args.input_file)
    batch_size = int(args.batch_size)

    label2idx_file = os.path.join(model_path, "label2ind.json")
    
    #---------------------------------------------------------
    print ("[INFO] step (2) check checkpoit directory and report file:")
    output_dir =  args.output_path
  
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    infer_out_file = os.path.join(output_dir, "classifier_pred.tsv")

    #-------------------------------------------------------
    print ("[INFO] step (3) load label to number dictionary:")
    lab2ind = json.load(open(label2idx_file))

    ind2label = {v: k for k,v in lab2ind.items()}

    
    print ("[INFO] model_path", model_path)
    print ("[INFO] max_seq_length", max_seq_length)
    print ("[INFO] batch_size", batch_size)
    #---------------------offensive-twitter-2019-task6-stB_emo_end_og_err.out----------------------------------
    print ("[INFO] step (4) Use defined funtion to extract tokanize data")


    if modelbased.lower()=="roberta":
        # tokenizer from pre-trained RoBERTa model
        print ("loading roberta setting")
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        
        print ("[INFO] step (5) Create an iterator of data with torch DataLoader.")
        
        infer_dataloader = regular_encode(input_file, tokenizer, lab2ind, False, batch_size=batch_size, maxlen = max_seq_length)
       
        model = RobertaForSequenceClassification.from_pretrained(model_path,num_labels=len(lab2ind))
        
    else:
        print ("[ERROR] please set modelbased to (bertweet or roberta)")

    #--------------------------------------
    print ("[INFO] step (6) run with parallel GPUs")
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            print("Run",modelbased, "with one GPU")
            model = model.to(device)
        else:
            n_gpu = torch.cuda.device_count()
            print("Run",modelbased, "with", n_gpu, "GPUs with max 4 GPUs")
            device_ids = GPUtil.getAvailable(limit = n_gpu)
            torch.backends.cudnn.benchmark = True
            model = model.to(device)
            model = nn.DataParallel(model, device_ids=device_ids)
    else:
        print("Run",modelbased, "with CPU")
        model = model
    #---------------------------------------------------
    print ("[INFO] step (7) set Parameters, schedules, and loss function:")
  
    num_infer_steps = len(infer_dataloader)
    print('num_infer_steps:', num_infer_steps)
    
    inference(model, infer_dataloader,infer_out_file, ind2label)
        


main_run(args)
