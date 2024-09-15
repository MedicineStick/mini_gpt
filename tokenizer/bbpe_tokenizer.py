
from tqdm import tqdm
from collections import Counter, defaultdict
import logging
import random
from collections import deque
import json
import csv
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def load_wiki_dict_v2()->list[str]:
    f = open("data/wiki.txt",mode='r')
    f2 = open("data/wiki.v2.txt",mode='w',encoding='utf8')
    lines = f.readlines()
    f.close()
    wiki_dict_list:list[str] = []

    for line in tqdm(lines):
        line = line.strip()
        if '=' in line:
            pass
        else:
            if len(line)>200:
                wiki_dict_list.append(line)
                f2.write(line.strip()+'\n')
    f2.close()
    return wiki_dict_list

def load_wiki_dict_v3()->list[str]:
    f = open("/home/tione/notebook/lskong2/projects/7.GPT2/data/wiki.v2.txt",mode='r')
    lines = f.readlines()
    wiki_dict_list:list[str] = []

    for line in tqdm(lines):
        line = line.strip()
        wiki_dict_list.append(line)
                
    return wiki_dict_list


def load_wiki_dict()->list[str]:
    f = open("/home/tione/notebook/lskong2/projects/7.GPT2/data/wiki.txt",mode='r')
    lines = f.readlines()

    i = 0
    last_title = ''
    last_article = []
    wiki_dict_list:list[str] = []
    repeat_title = []
    while i< len(lines):

        if i%10000 ==0:
            print('processed {} / {}'.format(i,len(lines)))
        line = lines[i].strip()
        if len(line)>=2:
           if line[0]=='=' and line[-1]=='=':
              if last_title!='' and len(last_article)>0:
                  wiki_dict_list.append(last_title+', '+' '.join(last_article))
                  last_title = line.replace('=','').strip()
                  last_article = []
              elif last_title=='':
                  last_title = line.replace('=','').strip()
                  last_article = []
              else:
                  pass  
           else:
              last_article.append(line.strip())
        else:
            pass
        i+=1
    print('wiki_dict_list: {}\n'.format(len(wiki_dict_list)))
    return wiki_dict_list

class byte_tranform:
    def __init__(self) -> None:
        pass
    def _byte_2_hex(self,byte_:int)->str:
        assert byte_<256 and byte_ >=0, f'invalid byte {byte_}'
        dp = hex(byte_)[2:]
        if len(dp)==1:
            return '0'+dp
        elif len(dp)==2:
            return dp
        else:
            return None
    def _str_2_bytes(self,str_:str)->bytes:
        return bytes(str_, 'utf-8')
    
    def _str_2_hexs(self,str_:str)->list[str]:
        return [ self._byte_2_hex(b) for b in self._str_2_bytes(str_)]
    
    def _hexs_2_str(self,hexs_:list[str])->str:
        try:
            return bytes.fromhex(''.join(hexs_)).decode('utf-8')
        except:
            logging.info(f" utf-8 codec can't decode byte {''.join(hexs_)}")
            return ""

    def _int_2_hex(self,num:int)->str:
        assert num<256 and num >=0, f'invalid byte {num}'
        return '{:02X}'.format(num).lower()

class bbpe_tokenizer:
    def __init__(
        self,
        lines_:list[str],
        max_tokens_:int,
        max_steps_:int,
        max_sents_:int,
        unk_token="<|endoftext|>",
        bos_token="<|beginoftext|>",
        eos_token="<|endoftext|>",
    ):
        logging.info("initialize bbpe tokenizer")
        self.corpus_lines = lines_
        self.tokens_lines = []
        self.max_tokens = max_tokens_
        self.BT = byte_tranform()
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_steps = max_steps_
        self.max_sents = max_sents_
        self.vocab = Counter()
        self.vocab_list = []

        
        self.bos_token_hex = ''.join(self.BT._str_2_hexs(self.bos_token))
        self.eos_token_hex = ''.join(self.BT._str_2_hexs(self.eos_token))
        self.vocab[self.bos_token_hex] = 100000000
        self.vocab[self.eos_token_hex] = 100000000

    def train(self):
        self._preprocess()
        del self.corpus_lines

        for i in range(0,self.max_steps):
            self._update_vocab(i)
            if len(self.vocab)>self.max_tokens:
                break
            self._update_corpus_tokens(i)

    def _preprocess(self):
        logging.info("preprocess corpus")
        temp = []
        count = 0
        for line in tqdm(self.corpus_lines):
            #line = self.bos_token+ line.strip() + self.eos_token
            line = line.strip()
            cur_hexs_line = self.BT._str_2_hexs(line)
            self.tokens_lines.append(cur_hexs_line)
            temp.extend(cur_hexs_line)
            count+=1
            if count > self.max_sents:
                break
        self.vocab = self.vocab+Counter(temp)

    ### concat all adjacent vocab into dict
    def _update_vocab(self,step:int):
        assert len(self.tokens_lines)>1,f'lenth of self.tokens_lines should great than 1'
        logging.info(f"updating vocabulary step{step}...")
        #while len(self.tokens_lines)==0:
        for tokens_line in tqdm(self.tokens_lines):
            for i in range(len(tokens_line) - 1):
                pair = tokens_line[i] + tokens_line[i + 1]
                self.vocab[pair] += 1

        if len(self.vocab)>self.max_tokens:
            top_items = self.vocab.most_common(self.max_tokens)
            self.vocab = Counter(dict(top_items))


        logging.info(f"lenth of vocab is {len(self.vocab)}...")
        #print(self.vocab)
    
    def _update_corpus_tokens(self,step:int)->list[str]:
        assert len(self.tokens_lines)>1,f'lenth of self.tokens_lines should great than 1'
        logging.info(f"updating corpus step{step}...")
        
        for i  in tqdm(range(0,len(self.tokens_lines))):
            tokens_line = self.tokens_lines[i]
            output_ = []
            s,e = 0,1
            max_key = ''  
            while e <= len(tokens_line):
                token = ''.join(tokens_line[s:e])
                if token in self.vocab:
                    max_key = token  
                    e += 1
                else:
                    if max_key:  
                        output_.append(max_key)
                        s = e - 1  
                        max_key = ''  
                    else:
                        s += 1  
                    e += 1
            self.tokens_lines[i] = output_
    
    def from_vocab_file(self,vocab_:str,vocab_size_:int,if_full_size:bool):
        logging.info("loading vocab...")
        self.vocab_list = []
        self.vocab = {}
        with open(vocab_,mode='r',encoding='utf8') as f1:
            lines = []
            if if_full_size:
                lines = f1.readlines()
            else:
                lines = f1.readlines()[0:vocab_size_]
            count = 0
            for token in lines:
                token = token.strip()
                self.vocab_list.append(token)
                self.vocab[token] = count
                count+=1
        self.vocab_list.append('00')
        self.vocab_list.append('00')

    def encode(self,sent:str,if_to_str:bool)->list:
        hex_list = []
        if if_to_str:
            output_list = [str(self.vocab[self.bos_token_hex])]
            hex_list.append(self.bos_token_hex)
        else:
            output_list = [int(self.vocab[self.bos_token_hex])]
            hex_list.append(self.bos_token_hex)
        hexs = self.BT._str_2_hexs(sent)
        print(hexs)
        hexsq = deque(hexs)
        while len(hexsq)>0:

            ts = []
            ts.append(hexsq.popleft())
            while len(hexsq)>0:
                ts.append(hexsq[0])
                t1 = ''.join(ts)
                if t1 in self.vocab:
                    hexsq.popleft()
                    continue
                else:
                    ts.pop()
                    break
            if if_to_str:
                output_list.append(str(self.vocab[''.join(ts)]))
                hex_list.append(''.join(ts))
            else:
                output_list.append(self.vocab[''.join(ts)])
                hex_list.append(''.join(ts))
        print(hex_list)
        return output_list
    
    def decode(self,encode_list:list[int])->str:
        #print(encode_list)
        #print(len( self.vocab_list))
        hexs = [ self.vocab_list[idx] for idx in encode_list]
        sents = self.BT._hexs_2_str(''.join(hexs))
        return sents

