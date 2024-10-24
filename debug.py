####test scaling law
from datasets import Dataset as Datasets
from tqdm import tqdm
import torch
def datasize_law(token_size: int) -> float:
    constant = 5.4 * (10**13)
    return (token_size / constant) ** -0.095

def param_law(param_size: int) -> float:
    constant = 8.8 * (10**13)
    return (param_size / constant) ** -0.076

def loop_dataset(name:str)->int:
    sqlite_url='sqlite:///./data/'+name+'.db'
    ds = Datasets.from_sql("states", sqlite_url)
    token_size = 0
    for index in tqdm(range(0,ds.num_rows)):
        token_size +=  len(ds[index]['data'].split())
    print("The token size of dataset {} is {}".format(name,token_size))
    return token_size


def test_tensor():
    padding_mask = torch.tensor([[True, True,False], [True,False, False]])
    padding_mask_expanded = padding_mask.unsqueeze(1)
    attention_mask = padding_mask_expanded & padding_mask_expanded.transpose(1, 2)
    print(attention_mask)
    attention_mask = attention_mask.unsqueeze(1).expand(-1, 1, -1, -1)
    print(attention_mask)
    print(padding_mask.shape)
    print(attention_mask.shape)
    exit(0)

if __name__ == "__main__":


    test_tensor()
    name = "train_corpus_c4_v2"
    #token_size = loop_dataset(name)
    # 1T 1,000,000,000,000
    #        1,964,434,092 
    #  0.002 T
    token_size = 1964434092
    print(datasize_law(token_size))

    print(param_law(90767360))