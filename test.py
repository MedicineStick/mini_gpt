
from model.gpt3 import GPT3,GPT3Config
import torch
from tqdm import tqdm
from datasets import Dataset as Datasets
from collections import Counter
from tokenizer.bbpe_tokenizer import bbpe_tokenizer
import os

def decode():
    device = None
    global_conf = GPT3Config("./conf/gpt3_v3.yaml")
    test_gpu = 3
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(test_gpu)
    else:
        device = torch.device("cpu")
    model_path = "./pt/pt_32l_0_00025_AdamW_wiki/model_iter_epoch_0_batch_250000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,test_gpu)
    gpt3.load_state_dict(torch.load(model_path),False)
    gpt3.eval()
    gpt3.to(device)

    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file('./vob/vocab.list.c4.v2',20833,True)
    prompt_list = ["The weather ",
                   "Lisa yesterday invented a new game which is called table tennis with hand ", 
                   "Today I am going to",
                   "Long bow ",
                   ]
    for prompt in prompt_list:
        token_list = bbpe.encode(prompt,False)
        print("prompt : ",prompt)
        print("prompt token_list: ",token_list)
        token_list = torch.tensor(token_list).unsqueeze(0).to(device)
        output = gpt3.inference(token_list,1)

        b,_ = output.shape

        for i in range(0,b):
            output_ = output[i].tolist()
            output_str = bbpe.decode(output_)
            print("response : ",output_str)


def decode1():
    device = None
    global_conf = GPT3Config("./conf/gpt3_v3.yaml")
    test_gpu = 4
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(test_gpu)
    else:
        device = torch.device("cpu")
    model_path = "./pt/pt_32l_0_00025_AdamW_wiki2/model_iter_epoch_0_batch_14000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,test_gpu)

    # Load the model's state_dict (weights) into the model
    gpt3.load_state_dict(torch.load(model_path, map_location=device))

    # Move the model to the correct device (if necessary)
    gpt3.to(device)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    prompt_list = ["The weather ",
                   "Lisa yesterday invented a new game which is called table tennis with hand ", 
                   "Today I am going to",
                   "Long bow ",
                   ]
    for prompt in prompt_list:
        token_list = enc.encode(prompt)
        print("prompt : ",prompt)
        print("prompt token_list: ",token_list)
        token_list = torch.tensor(token_list).unsqueeze(0).to(device)
        output = gpt3.inference(token_list,1)

        b,_ = output.shape

        for i in range(0,b):
            output_ = output[i].tolist()
            output_str = enc.decode(output_)
            print("response : ",output_str)

def test_dataset():
    
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



def test_token():
    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file('./data/vocab.list.c4.v2',11255,True)
    str1 = "On July 16 , 2015 , the series was nominated for seven Primetime Emmy Awards , including Outstanding Comedy Series ."
    list1 = bbpe.encode(str1,False)
    str2 = bbpe.decode(list1)
    print(str1)
    print(list1)
    print(str2)

    print("---------")

    token1 = "0 542 89 210 31 47 258 120 207 97 76 29 25 17 27 91 213 107 209 83 65 20 45 77 31 1208 123 220 34 38 78 107 120 144 95 54 59 50 25 233 590 54 121 76 181 175 228 203 13521 3228 58 20 133 103 42 14 21 278 39 169 34 121 117 14 42 17 27 183 203 2104 216 110 118 68 34 201 76 19 87 59 200 103 949 20770 1"
    token_list = [int(num) for num in token1.split()]
    
    str2 = bbpe.decode(token_list)
    print(str2)


def test_tiktoken():
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    #assert enc.decode(enc.encode("hello world")) == "hello world"

    print(enc.encode("Lisa yesterday invented a new game which is called table tennis with hand "))

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4o")

    exit(0)

if __name__ =="__main__":

    #test_tiktoken()

    #test_token()

    decode1()
