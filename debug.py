####test scaling law
from datasets import Dataset as Datasets
from tqdm import tqdm
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


if __name__ == "__main__":
    name = "train_corpus_c4_v2"
    #token_size = loop_dataset(name)
    token_size = 1964434092
    print(datasize_law(token_size))

    print(param_law(90767360))