
from tqdm import tqdm
from collections import Counter, defaultdict
import logging
from collections import deque
import random
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

    def encode(self,sent:str,if_to_str:bool)->list:
        output_list = []
        hexs = self.BT._str_2_hexs(sent)
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
            else:
                output_list.append(self.vocab[''.join(ts)])
        return output_list
    
    def decode(self,encode_list:list[int])->str:
        hexs = [ self.vocab_list[idx] for idx in encode_list]
        sents = self.BT._hexs_2_str(''.join(hexs))
        return sents

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

def new_vocab():
    BT =byte_tranform()
    set1 = set()
    f2 = open('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.final.list.v3',mode='w',encoding='utf8') 
    

    with open('/home/tione/notebook/lskong2/projects/7.GPT2/data/vocab.list.v3',mode='r',encoding='utf8') as f1:
        lines = f1.readlines()[:20017]
        for line in lines:
            token,_ = line.strip().split('\t')
            set1.add(token.strip())
            f2.write(token+'\n')
    for i in range(0,256):
        c = (BT._int_2_hex(i).strip())
        if c not in set1:
            f2.write(c+'\n')
        

if __name__  == "__main__":
   new_vocab()
   encode_corpus_v3()
   
   
   