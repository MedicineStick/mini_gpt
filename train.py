from dataset_gpt3 import DatasetGPT3,process_db
import csv
from gpt3 import GPT3,GPT3Config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
device = None
from thop import profile
from bbpe_tokenizer import bbpe_tokenizer



def train(name:str):
    dataset = DatasetGPT3(
        sqlite_url='sqlite:///./data/'+name+'.db'
        )
    global_conf = GPT3Config("/home/tione/notebook/lskong2/projects/7.GPT2/conf/gpt3_v3.yaml")
    
    if global_conf.if_gpu==1:
        device = torch.device("cuda")
        torch.cuda.set_device(global_conf.device_id)
    else:
        device = torch.device("cpu")

    if os.path.exists(global_conf.output_path):
        pass
    else:
        os.makedirs(global_conf.output_path)
    
    gpt3 = GPT3(global_conf,device)
    if global_conf.pretrain_model == "":
        pass
    else:
        gpt3.load_state_dict(torch.load(global_conf.pretrain_model))
    gpt3.to(device)
    bbpe = bbpe_tokenizer([],0,0,0)
    bbpe.from_vocab_file(global_conf.wlist,global_conf.wlist_size,True)


    #optimizer = optim.SGD(gpt3.parameters(), lr=global_conf.learning_rate, momentum=0.9, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(gpt3.parameters(), lr=global_conf.learning_rate)
    optimizer = torch.optim.AdamW(gpt3.parameters(), lr=global_conf.learning_rate)
    def collate_fn(tensor_list):

        max_length = max(tensor.size(0) for tensor in tensor_list)
        padded_tensors = [torch.cat([tensor, torch.full((max_length - tensor.size(0),), global_conf.vocab_size, dtype=tensor.dtype)]) 
                      for tensor in tensor_list]
        padded_tensor_batch = torch.stack(padded_tensors)
        
        return padded_tensor_batch
    
    dataloader = DataLoader(
        dataset, 
        batch_size=global_conf.n_batch_size, 
        shuffle=True,
        collate_fn=collate_fn
        )
    log_file = open(global_conf.output_path+'/log.txt1',mode='w')
    #tensor = (torch.ones((10, 3000),dtype=torch.int32).to(device),)
    #flops, params = profile(gpt3, inputs=tensor)
    #print('FLOPs =', flops)
    #print('params =', params)
    total_batches = len(dataloader)
    for epoch in range(0,global_conf.n_epoch):
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            logits_reshaped,labels_reshaped = gpt3(data)
            
            """label = data[0].cpu().tolist()
            logit = torch.argmax(logits_reshaped,dim=-1).cpu().tolist()
            preds = bbpe.decode(logit)
            labels = bbpe.decode(label)
            print(preds)
            print(labels) """
            

            loss = gpt3.loss(logits_reshaped, labels_reshaped)
            
            loss.backward()
            optimizer.step()
            print("batch_idx {}/{}, loss{}".format(batch_idx,total_batches,loss.item()))
            log_file.write(f'Epoch [{epoch + 1}/{global_conf.n_epoch}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f} '+'\n')
            log_file.flush()

            if batch_idx%global_conf.save_per_batchs ==0:
                torch.save(gpt3.state_dict(), global_conf.output_path+'/model_iter_'+str(epoch)+'_batch_'+str(batch_idx)+'.pth')

        torch.save(gpt3.state_dict(), global_conf.output_path+'/model_iter_'+str(epoch)+'.pth')
    log_file.close()
def test(name):
    print(1)
    dataset = DatasetGPT3(
        sqlite_url='sqlite:///./data/'+name+'.db'
        )
    max_ = 0
    count_dict = {500:0,1000:0,1500:0,2000:0,2500:0,3000:0,3500:0,4000:0}
    for index in tqdm(range(0,dataset.__len__())):
        length = len(dataset.ds[index]['data'].split())
        
        if length <500:
            count_dict[500]+=1
        elif length <1000:
            count_dict[1000]+=1
        elif length <1500:
            count_dict[1500]+=1
        elif length <2000:
            count_dict[2000]+=1
        elif length <2500:
            count_dict[2500]+=1
        elif length <3000:
            count_dict[3000]+=1
        elif length <3500:
            count_dict[3500]+=1
        elif length <4000:
            count_dict[4000]+=1

    print(max_)    
    
    print(count_dict)

    for k,v in count_dict.items():
        print(k,v*100/dataset.__len__())
    print(2)

if __name__ =="__main__":
    name =  'train_corpus_v3'
    """with open('./data/'+name+'.csv',mode='w', newline='') as f1:
        spamwriter = csv.writer(f1)
        corpus_lines  = [[1,2],[4,5,6,7]]
        spamwriter.writerow(['data'])
        for line in corpus_lines:
            tokens = ' '.join([ str(id) for id in line])
            spamwriter.writerow([tokens])
    csv_file='./data/'+name+'.csv'
    db_path='./data/'+name+'.db'
    process_db(csv_file,db_path)"""

    train(name)
    #test(name)