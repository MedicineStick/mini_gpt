
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
    lines = f.readlines()
    f.close()
    wiki_dict_list:list[str] = []

    for line in tqdm(lines):
        line = line.strip()
        if '=' in line:
            pass
        else:
            if len(line)>2:
                wiki_dict_list.append(line)
    return wiki_dict_list


def load_wiki_dict()->list[str]:
    f = open("data/wiki.txt",mode='r')
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
        return bytes.fromhex(''.join(hexs_)).decode('utf-8')

    def _int_2_hex(self,num:int)->str:
        assert num<256 and num >=0, f'invalid byte {num}'
        return '{:02X}'.format(num).lower()

class bbpe_tokenizer:
    def __init__(
        self,
        lines_:list[str],
        max_tokens_:int,
        max_steps_:int,
        unk_token="<|endoftext|>",
        bos_token="<|beginoftext|>",
        eos_token="<|endoftext|>",
    ):
        logging.info("initialize bbpe tokenizer")
        self.corpus_lines = lines_
        self.tokens_lines = deque()
        self.max_tokens = max_tokens_
        self.BT = byte_tranform()
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_steps = max_steps_
        self.vocab = {}
        self.vocab_list = []
            
    def _reset_token_lists(self,tokens_lines_:list[str]):
        self.tokens_lines = tokens_lines_

    def train(self):
        self._preprocess()
        self.corpus_lines = []
        vocab = Counter(self.tokens_lines)
        self.vocab =  {word: freq for word, freq in vocab.items()}
        for i in range(0,self.max_steps):
            self._update_vocab(i)
            self._reset_token_lists(self._update_corpus_tokens(i))

    def _preprocess(self):
        logging.info("preprocess corpus")
        for line in tqdm(self.corpus_lines):
            line = line.strip() + ' ' + self.eos_token
            self.tokens_lines.extend(self.BT._str_2_hexs(line))

    ### concat all adjacent vocab into dict
    def _update_vocab(self,step:int):
        assert len(self.tokens_lines)>1,f'lenth of self.tokens_lines should great than 1'
        logging.info(f"updating vocabulary step{step}...")
        #while len(self.tokens_lines)==0:
        temp_tokens_lines_= deque()
        while len(self.tokens_lines)>1:
            t1,t2 = self.tokens_lines.popleft(),self.tokens_lines[0]
            temp_tokens_lines_.append(t1)
            vocab_ = ''.join(t1+t2)
            if vocab_ not in self.vocab.keys():
               self.vocab[vocab_] = 1
            else:
               self.vocab[vocab_] += 1
        temp_tokens_lines_.append(self.tokens_lines.popleft())
        self.tokens_lines = temp_tokens_lines_
        self.vocab = dict(sorted(
            self.vocab.items(), key=lambda x: x[1], reverse=True
            )[:min(self.max_tokens,len(self.vocab))])
        logging.info(f"lenth of vocab is {len(self.vocab)}...")
        #print(self.vocab)
    
    def _update_corpus_tokens(self,step:int)->list[str]:
        assert len(self.tokens_lines)>1,f'lenth of self.tokens_lines should great than 1'
        logging.info(f"updating corpus step{step}...")
        new_tokens_lines_ = deque()
        last_add = False
        while len(self.tokens_lines)>1:
        #for i in tqdm(range(0,len(self.tokens_lines)-1)):
            t1,t2 = self.tokens_lines.popleft(),self.tokens_lines[0]
            vocab = ''.join(t1+t2)
            if vocab not in self.vocab.keys():
               if last_add:
                   last_add = False
               else:
                   new_tokens_lines_.append(t1)
                   last_add = False
                   if len(self.tokens_lines)==1:
                      new_tokens_lines_.append(t2)
            else:
               if last_add:
                   last_add = False
               else:
                   new_tokens_lines_.append(vocab)
                   last_add = True
        return new_tokens_lines_
    
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
                token= token.strip()
                self.vocab_list.append(token)
                self.vocab[token] = count
                count+=1

    def encode(self,sent:str,if_to_str:bool)->list:
        #logging.info("encoding string...")
        output_list = []
        sent = sent.strip() + ' ' #+ self.eos_token
        hexs = self.BT._str_2_hexs(sent)
        hexsq = deque(hexs)
        while len(hexsq)>0:

            ts = []
            ts.append(hexsq.popleft())
            while len(hexsq)>0:
                ts.append(hexsq[0])
                t1 = ''.join(ts)
                if t1 in self.vocab.keys():
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
    debug = False

    if debug:
 
        str1 = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n . It met with positive sales in Japan , and was praised by both Japanese and western critics . After release , it received downloadable content , along with an expanded edition in November of that year . It was also adapted into manga and an original video animation series . Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 . Media.Vision would return to the franchise with the development of Valkyria : Azure Revolution for the PlayStation 4 ."
        max_tokens = 30
        max_steps = 4
        bbpe = bbpe_tokenizer([str1],max_tokens,max_steps)
        bbpe.train()
        s4 = ''.join(bbpe.tokens_lines)
        BT = byte_tranform()
        s3 = BT._hexs_2_str(s4)
        print(str1)
        print(s3)

        exit(0)
        dict1 = {'a':11,'b':1,'c':1,'d':1}
        top_500_items = sorted(dict1.items(), key=lambda x: x[1], reverse=True)[:4]
        print(top_500_items)
        exit(0)

    logging.info("load_wiki_dict")
    wikitexts = load_wiki_dict()#   [:2000]
    max_tokens = 20000
    max_steps = 10
    bbpe = bbpe_tokenizer(wikitexts,max_tokens,max_steps)
    bbpe.train()

    with open('./data/vocab.list',mode='w',encoding='utf8') as f1:
        for k,v in bbpe.vocab.items():
            f1.write(k+'\t'+str(v)+'\n')


def main_test():
    max_tokens = 20000
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

def new_vocab():
    BT =byte_tranform()
    set1 = set()
    
    for i in range(0,256):
        set1.add(BT._int_2_hex(i).strip())
    with open('./data/vocab.list',mode='r',encoding='utf8') as f1:
        lines = f1.readlines()[:11255]
        for line in lines:
            token,_ = line.strip().split('\t')
            set1.add(token.strip())
    list1 = list(set1)
    random.shuffle(list1)

    with open('./data/vocab.final.list',mode='w',encoding='utf8') as f2:
        for line in list1:
            f2.write(line+'\n')
        

if __name__  == "__main__":
   encode_corpus_v2()
   
   