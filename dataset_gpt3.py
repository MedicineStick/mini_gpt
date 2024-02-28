import torch
from torch.utils.data import DataLoader, Dataset
import sqlite3
from datasets import Dataset as Datasets
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')




def process_db(csv_file:str,db_path:str):
    logging.info('connecting to db...')
    conn = sqlite3.connect(db_path)
    logging.info('read csv to db...')
    df = pd.read_csv(csv_file)
    logging.info('to sql...')
    df.to_sql("states", conn, if_exists="replace")

class DatasetGPT3(Dataset):
    def __init__(
        self,
        sqlite_url:str,
        transform = None,
        device = 'gpu',
        if_shuffle = True,
        ):
        super().__init__()

        #uri = "sqlite:///"+db_path
        self.ds = Datasets.from_sql("states", sqlite_url)
        if if_shuffle:
            self.ds.shuffle()
    def __len__(self):
        return self.ds.num_rows
    
    def __getitem__(self, index):
        number_list = [int(num) for num in self.ds[index]['data'].split()]
        result_tensor = torch.tensor(number_list)
        return result_tensor
        
        