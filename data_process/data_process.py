from tqdm import tqdm
from collections import Counter, defaultdict
import logging
import random
from collections import deque
import json
import csv
from bbpe_tokenizer import bbpe_tokenizer,load_wiki_dict,load_wiki_dict_v2,load_wiki_dict_v3,byte_tranform

from dataset_gpt3 import process_db



def main_train():
    debug = True

    if debug:
        #load_wiki_dict_v2()
        bbpe = bbpe_tokenizer(None,1,1,1)
        out = bbpe.BT._hexs_2_str(['e2'])
        print(out)
        out = bbpe.BT._str_2_hexs('—')
        print(out)
        #exit(0) 
        f2 = open('./data/vocab.list.v2.decode',mode='w',encoding='utf8')
        with open('./data/vocab.list.v2',mode='r',encoding='utf8') as f1:
            lines = f1.readlines()
            for line in tqdm(lines):
                token, idx = line.strip().split('\t')
                token = bbpe.BT._hexs_2_str([token.strip()])
                f2.write(token+'\t'+idx+'\n')
        exit(0)
        

    logging.info("load_wiki_dict")
    wikitexts = load_wiki_dict_v3()#   [:2000]
    max_tokens = 40000
    max_steps = 10
    max_sents = 100000
    bbpe = bbpe_tokenizer(wikitexts,max_tokens,max_steps,max_sents)
    bbpe.train()

    with open('./data/vocab.list.v2',mode='w',encoding='utf8') as f1:
        for k,v in bbpe.vocab.items():
            f1.write(k+'\t'+str(v)+'\n')



def main_train_c4_v2():
    file = open("data/train_corpus_c4_v1.txt",mode='r')
    c4_texts = file.readlines()
    random.shuffle(c4_texts)

    max_tokens = 40000
    max_steps = 10
    max_sents = 500000
    bbpe = bbpe_tokenizer(c4_texts,max_tokens,max_steps,max_sents)
    bbpe.train()

    with open('./data/vocab.list.c4.v1',mode='w',encoding='utf8') as f1:
        for k,v in bbpe.vocab.items():
            f1.write(k+'\t'+str(v)+'\n')

def main_train_c4_v1():
    file = open("data/train_corpus_c4_v1.txt",mode='r')
    c4_texts = file.readlines()
    random.shuffle(c4_texts)

    max_tokens = 40000
    max_steps = 10
    max_sents = 500000
    bbpe = bbpe_tokenizer(c4_texts,max_tokens,max_steps,max_sents)
    bbpe.train()

    with open('./data/vocab.list.c4.v1',mode='w',encoding='utf8') as f1:
        for k,v in bbpe.vocab.items():
            f1.write(k+'\t'+str(v)+'\n')

def main_train_c4():
    logging.info("load_c4")

    with open('/home/tione/notebook/lskong2/projects/7.GPT/data/train_corpus_c4_v1.txt',mode='w', ) as f1:
        for i in range(0,5):

            file = "/home/tione/notebook/lskong2/data/c4/c4-train.0000"+str(i)+"-of-01024.json"
            f = open(file,mode='r')
            lines = f.readlines()
            for line in tqdm(lines):
                output = json.loads(line.strip())
                str_list = output['text'].strip().split('\n')
                for str_ in str_list:
                    if len(str_)>10 and 'http' not in str_:
                        try:
                            f1.write(str_+'\n')
                        except Exception:
                            print(str_)
                            exit(0)


def main_test():
    max_tokens = 100000
    max_steps = 10
    str1 = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 ."
    bbpe = bbpe_tokenizer([],max_tokens,max_steps)
    bbpe.from_vocab_file('./data/vocab.list',11255,False) #11255
    str1list = bbpe.encode(str1)
    str2 = bbpe.decode(str1list)
    print(str1)
    print(str1list)
    print(str2)

def encode_corpus():
    max_tokens = 20000
    max_steps = 10
    corpus_lines = load_wiki_dict()
    bbpe = bbpe_tokenizer([],max_tokens,max_steps)
    bbpe.from_vocab_file('./data/vocab.final.list',11255,True)
    logging.info("encoding string...")
    with open('./data/train_corpus_v1.csv',mode='w', newline='') as f1:
        spamwriter = csv.writer(f1)
        spamwriter.writerow(['data'])
        for line in tqdm(corpus_lines):
            line = line.strip() + ' ' + bbpe.eos_token
            str1list = bbpe.encode(line,False)
            tokens = ' '.join([ str(id) for id in str1list])
            spamwriter.writerow([tokens])

def encode_corpus_v2():
    max_tokens = 20000
    max_steps = 10
    corpus_lines = load_wiki_dict_v2()
    bbpe = bbpe_tokenizer([],max_tokens,max_steps)
    bbpe.from_vocab_file('./data/vocab.final.list',11255,True)
    logging.info("encoding string...")
    with open('./data/train_corpus_v2.csv',mode='w', newline='') as f1:
        spamwriter = csv.writer(f1)
        spamwriter.writerow(['data'])
        for line in tqdm(corpus_lines):
            line = bbpe.bos_token+' '+line.strip() + ' ' + bbpe.eos_token
            str1list = bbpe.encode(line,False)
            tokens = ' '.join([ str(id) for id in str1list])
            spamwriter.writerow([tokens])

def encode_corpus_v3():
    max_tokens = 20017
    max_steps = 10
    corpus_lines = load_wiki_dict_v3()
    bbpe = bbpe_tokenizer([],max_tokens,max_steps,0)
    bbpe.from_vocab_file('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.final.list.v3',max_tokens,True)

    str1 = "27 518 1622 6925 162 62 1601 1243 1643 177 461"
    list1 = [int(i)  for i in str1.strip().split()]
    output = bbpe.decode(list1)
    print(output)
    #return output

    logging.info("encoding string...")
    with open('/home/tione/notebook/lskong2/projects/7.GPT2/data/train_corpus_v3.csv',mode='w', newline='') as f1:
        spamwriter = csv.writer(f1)
        spamwriter.writerow(['data'])
        b = "3c7c626567696e6f66746578747c3e"
        e = "3c7c656e646f66746578747c3e"
        count = 0
        for line in tqdm(corpus_lines):
            #line = bbpe.bos_token+line.strip()+bbpe.eos_token
            str1list = bbpe.encode(line,False)
            str1list.insert(0, bbpe.vocab[bbpe.bos_token_hex])
            str1list.append(bbpe.vocab[bbpe.eos_token_hex])
            tokens = ' '.join([ str(id) for id in str1list])
            spamwriter.writerow([tokens])

def encode_corpus_v4():
    max_tokens = 20017
    max_steps = 10
    file = open("data/train_corpus_c4_v1.txt",mode='r')
    corpus_lines = file.readlines()
    random.shuffle(corpus_lines)
    bbpe = bbpe_tokenizer([],max_tokens,max_steps,0)
    bbpe.from_vocab_file('./data/vocab.list.c4.v2',max_tokens,True)
    logging.info("encoding string...")
    with open('./data/train_corpus_c4_v2.csv',mode='w', newline='') as f1:
        spamwriter = csv.writer(f1)
        spamwriter.writerow(['data'])
        for line in tqdm(corpus_lines):
            #line = bbpe.bos_token+line.strip()+bbpe.eos_token
            str1list = bbpe.encode(line,False)
            str1list.insert(0, bbpe.vocab[bbpe.bos_token_hex])
            str1list.append(bbpe.vocab[bbpe.eos_token_hex])
            tokens = ' '.join([ str(id) for id in str1list])
            spamwriter.writerow([tokens])


def new_vocab():
    BT =byte_tranform()
    set1 = set()
    f2 = open('/home/tione/notebook/lskong2/projects/7.GPT/data/vocab.list.c4.v2',mode='w',encoding='utf8') 
    

    with open('/home/tione/notebook/lskong2/projects/7.GPT/data/vocab.list.c4.v1',mode='r',encoding='utf8') as f1:
        lines = f1.readlines()[:20762]
        for line in lines:
            token,_ = line.strip().split('\t')
            set1.add(token.strip())
            f2.write(token+'\n')
    for i in range(0,256):
        c = (BT._int_2_hex(i).strip())
        if c not in set1:
            f2.write(c+'\n')
        

if __name__  == "__main__":
   #new_vocab()
   #encode_corpus_v3()
   csv_file = "data/train_corpus_c4_v2.csv"
   db_path = "data/train_corpus_c4_v2.db"
   process_db(csv_file,db_path)
   
   
   