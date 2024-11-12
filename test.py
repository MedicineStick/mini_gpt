
from model.gpt3 import GPT3,GPT3Config
import torch
from tqdm import tqdm
device = None
from datasets import Dataset as Datasets
from collections import Counter
from tokenizer.bbpe_tokenizer import bbpe_tokenizer


def decode():

    global_conf = GPT3Config("./conf/gpt3_v3.yaml")
    test_gpu = 5
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(test_gpu)
    else:
        device = torch.device("cpu")
    model_path = "./pt/pt_32l_0_00025_AdamW_c4_v10/model_iter_epoch_0_batch_4000.pth"
    global_conf.if_train = False
    gpt3 = GPT3(global_conf,device)
    gpt3.load_state_dict(torch.load(model_path),False)
    gpt3.eval()
    gpt3.to(device)

    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file('./data/vocab.list.c4.v2',20833,True)
    prompt_list = ["The weather ",
                   "How do you ", 
                   "Today I plan to ",
                   "When you are eating you should ",
                   "When you are swimming you should ",
                   "Output the ",
                   "I want to ",
                   "I am going to "
                   ]
    for prompt in prompt_list:
        token_list = bbpe.encode(prompt,False)
        token_list = torch.tensor(token_list).unsqueeze(0).to(device)
        output = gpt3.inference(token_list,1)

        b,_ = output.shape

        for i in range(0,b):
            output_ = output[i].tolist()
            output_str = bbpe.decode(output_)
            print(output_)
            print(output_str+'\n')
    

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

if __name__ =="__main__":


    #test_token()

    decode()
