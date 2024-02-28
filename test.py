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

from bbpe_tokenizer import bbpe_tokenizer


def train():

    global_conf = GPT3Config("/home/tione/notebook/lskong2/projects/7.GPT2/conf/gpt3_v1.yaml")
    
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(global_conf.device_id)
    else:
        device = torch.device("cpu")
    model_path = "/home/tione/notebook/lskong2/projects/7.GPT2/pt_12l_0_00025_AdamW_v2/model_iter_0_batch_10000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,device)
    gpt3.load_state_dict(torch.load(model_path),False)
    gpt3.eval()
    gpt3.to(device)

    bbpe = bbpe_tokenizer([],0,0)
    bbpe.from_vocab_file('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.final.list',11255,True)
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

        b,l = output.shape

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



if __name__ =="__main__":


    #test_v1()

    train()
