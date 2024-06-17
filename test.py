from dataset_gpt3 import DatasetGPT3,process_db
import csv
from gpt3 import GPT3,GPT3Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
device = None
from thop import profile
from datasets import Dataset as Datasets
from collections import Counter, defaultdict
from bbpe_tokenizer import bbpe_tokenizer


def train():

    global_conf = GPT3Config("/home/tione/notebook/lskong2/projects/7.GPT2/conf/gpt3_v3.yaml")
    
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(global_conf.device_id)
    else:
        device = torch.device("cpu")
    model_path = "/home/tione/notebook/lskong2/projects/7.GPT2/pt_12l_0_00025_AdamW_v5/model_iter_3_batch_42000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,device)
    gpt3.load_state_dict(torch.load(model_path),False)
    gpt3.eval()
    gpt3.to(device)

    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.final.list.v3',11255,True)
    prompt_list = ["The weather ",
                   "how do you", 
                   "today ",
                   "If these steps ",
                   "For a ",
                   "Verify that",
                   "outputs the",
                   ]
    for prompt in prompt_list:
        token_list = bbpe.encode(prompt,False)
        token_list = torch.tensor(token_list).unsqueeze(0).to(device)
        output = gpt3.inference(token_list)

        b,_ = output.shape

        for i in range(0,b):
            output_ = output[i].tolist()
            output_str = bbpe.decode(output_)
            print(output_str+'\n')
    




    def collate_fn(tensor_list):

        max_length = max(tensor.size(0) for tensor in tensor_list)
        padded_tensors = [torch.cat([tensor, torch.full((max_length - tensor.size(0),), global_conf.vocab_size, dtype=tensor.dtype)]) 
                      for tensor in tensor_list]
        padded_tensor_batch = torch.stack(padded_tensors)
        
        return padded_tensor_batch
    

def test_v1():
    tensor_in = torch.randn((2,2,5))
    print(tensor_in)
    mean = tensor_in.mean(dim=-1, keepdim=True)
    print(mean)
    std = tensor_in.std(dim=-1, keepdim=True, unbiased=False)
    print(std)

def test_v2():
    
    ds = Datasets.from_sql("states", 'sqlite:///./data/train_corpus_v3.db')
    counter = Counter()
    total = 0
    for index in tqdm(range(0,ds.num_rows)):
        nums = ds[index]['data'].split()
        total+=len(nums)
        for num in nums:
            counter[num]+=1

    for k,v in counter.items():
        print(k,v)

        
def test_v3():
    mask = torch.triu(torch.ones(3, 3) * float('-inf'), diagonal=1)
    att_score = torch.randn((1,3,3))
    print("att_score",att_score)
    print("mask",mask)
    wv = torch.randn((1,3,5))
    att_score = att_score + mask[None, :, :] 
    
    att_score = nn.functional.softmax(att_score,dim=-1)
    print("att_score",att_score)
    att = torch.matmul(att_score,wv)
    print("wv",wv)
    print("att",att)


def test_v4():
    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.final.list.v3',11255,True)
    str1 = "On July 16 , 2015 , the series was nominated for seven Primetime Emmy Awards , including Outstanding Comedy Series . Fey herself was nominated both as the creator / executive producer of the series and for Outstanding Guest Actress in a Comedy Series for her guest performance as Marcia , a bumbling prosecutor in reference to Marcia Clark ."
    list1 = bbpe.encode(str1,False)
    print(list1)
    str2 = bbpe.decode(list1)

    token1 = "27 573 73 94 312 24 162 46 116 12 41 17 12 221 169 23 118 701 92 43 242 42 15 118 24 150 38 23 124 79 85 31 44 386 201 301 31 20 189 150 111 998 49 48 51 468 40 21 45 14 62 76 78 176 29 117 634 52 32 34 44 118 103 72 34 43 187 180 54 56 211 32 70 311 205 374 55 91 15 41 170 17 67 179 60 54 80 86 320 224 58 126 66 49 129 38 69 34 12 51 14 240 446 91 58 14 90 32 55 12 188 116 70 18 20 320 79 58 194 54 113 57 34 311 177 77 70 63 135 223 83 274 143 164 29 24 12 193 1243 137 272 56 127 94 15 284 128 147 15 102 257 15 128 34 109 99 311 60 309 72 164 47 15 124 313 363 47 155 195 52 113 103 43 24 72 167 63 119 135 497 289 40 48 94 111 47 32 70 14 148 313 363 14 62 114 126 54 15 290 78 14 62 170 17 59 31 20 17 31 38 17 29 100 61 6505 216 15 320 12 63 47 181 194 92 224 78 72 389 227 56 146 15 66 21 289 62 90 15 76 104 193 49 20 130 59 79 264 57 34 21 45 126 183 67 128 142 130 260 40 14 18 211 73 29 154 57 34 60 417 161 34 67 51 166 34 81 477 46 95 23 71 278 46 116 12 274 186 18242 2935 277 90 15 104 59 42 47 32 61 278 89 39 165 36 46 52 42 56 28"
    token_list = [int(num) for num in token1.split()]

    global_conf = GPT3Config("/home/tione/notebook/lskong2/projects/7.GPT2/conf/gpt3_v3.yaml")
    
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(global_conf.device_id)
    else:
        device = torch.device("cpu")
    model_path = "/home/tione/notebook/lskong2/projects/7.GPT2/pt_12l_0_00025_AdamW_v3/model_iter_3_batch_30000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,device)
    gpt3.load_state_dict(torch.load(model_path))
    gpt3.to(device)
    data = torch.tensor(token_list).unsqueeze(0).to(device)
    data = data.to(device)
    logits_reshaped,labels_reshaped = gpt3(data)
    loss = gpt3.loss(logits_reshaped, labels_reshaped)
    loss = loss.item()


    


    print(str1)
    print(str2)

if __name__ =="__main__":


    #test_v4()

    train()
