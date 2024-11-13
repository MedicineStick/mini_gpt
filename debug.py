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

def attn():
    import torch

    scale = 0.125

    # 5 tokens, 1 padding
    seq_length = 5
    max_seq_length = 10

    q = torch.randn(1, 2, seq_length + 1, 8)
    k = torch.randn(1, 2, max_seq_length, 8)
    v = torch.randn(1, 2, max_seq_length, 8)
    mask = torch.tensor([[[
        [True, False, False, False, False, False, False, False, False, False],  # regular mask
        [True, True, False, False, False, False, False, False, False, False],
        [True, True, True, False, False, False, False, False, False, False],
        [True, True, True, True, False, False, False, False, False, False],
        [True, True, True, True, True, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False]]]])  # padding mask


    def torch_sdpa(q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0, scale=scale)


    def naive_sdpa_1(q, k, v, mask):
        att = (q @ k.transpose(-2, -1)) * scale
        att = torch.where(~mask, float("-inf"), att)
        att = torch.nn.functional.softmax(att, dim=-1)
        return att @ v


    def naive_sdpa_2(q, k, v, mask):
        att = (q @ k.transpose(-2, -1)) * scale
        att = torch.where(~mask, torch.finfo(att.dtype).min, att)
        att = torch.nn.functional.softmax(att, dim=-1)
        return att @ v


    y = torch_sdpa(q, k, v, mask)
    print(torch.isnan(y).any())

    y = naive_sdpa_1(q, k, v, mask)
    print(torch.isnan(y).any())

    y = naive_sdpa_2(q, k, v, mask)
    print(torch.isnan(y).any())

    exit(0)

def test_mask():
    import torch

    scale = 0.125

    # 5 tokens, 1 padding
    seq_length = 5
    max_seq_length = 10

    q = torch.randn(1, 2, seq_length + 1, 8)
    k = torch.randn(1, 2, max_seq_length, 8)
    v = torch.randn(1, 2, max_seq_length, 8)

    original_mask = torch.tensor([[[
    [True, False, False],  # causal mask
    [True, True, False],
    [False, False, False],
    [False, False, False],
    ]]])  # padding mask

    original_mask = original_mask.expand(2,2,-1,-1)

    attention_mask1 = (1.0 - original_mask.to(dtype=torch.float16)) * -10000.0
    print(attention_mask1)
    attention_mask = torch.full(original_mask.shape, float("-inf"))

    # Set True values to 0.0
    attention_mask[original_mask] = 0.0

    # Set rows with all False to 0.0
    all_false_rows = ~original_mask.any(dim=-1)  # Identify rows with all False
    attention_mask[all_false_rows] = 0.0  # Set those rows to 0.0

    print(attention_mask)
    print(attention_mask.shape)
    exit(0)


def test_mask1():
    fill_value = -float("inf") 
    row0 = torch.randn(4)
    row1 = torch.tensor([fill_value for _ in range(4)])
    matrix = torch.stack([row0, row1]).requires_grad_(True)
    matrix = torch.nan_to_num(matrix,0.0)
    out = torch.softmax(matrix, 1)
    
    print(out)
    exit(0)

def test_shape():

    A = torch.rand(1, 3, 3,1)
    print(A)
    B = A.transpose(1, 2).contiguous()
    print(B)
    C = A.permute(0,2,1,3).contiguous()
    print(C)


    exit(0)

def check_vob():

    vob = "vob/vocab.list.c4.v2"
    f = open(vob,mode='r',encoding='utf8')
    vob_set = set()
    lines =f.readlines()
    f.close()

    for line in lines:
        line = line.strip()
        if line in vob_set:
            print(line)
        else:
            vob_set.add(line)
    exit(0)

def check_loss():
    import torch.nn as nn
    logit =  torch.load("./temp/logit.pt", weights_only=True).cpu()
    label =  torch.load("./temp/label.pt", weights_only=True).cpu()
    #logit = torch.Tensor([[1.0,3.0,4.0]])
    #label = torch.Tensor([2]).type(dtype=torch.LongTensor)
    print(f"logit.shape {logit.shape}")
    print(f"label.shape {label.shape}")
    print(label.unique())
    print(logit.max(), logit.min())

    # logit shape (16352, 20834)
    # label shape (16352)
    lossf = nn.CrossEntropyLoss(ignore_index=20833)

    loss1 = lossf(logit,label)

    print(loss1)
    exit(0)


if __name__ == "__main__":
    check_loss()
    check_vob()
    test_shape()
    test_tensor()
    name = "train_corpus_c4_v2"
    #token_size = loop_dataset(name)
    # 1T 1,000,000,000,000
    #        1,964,434,092 
    #  0.002 T
    token_size = 1964434092
    print(datasize_law(token_size))

    print(param_law(90767360))