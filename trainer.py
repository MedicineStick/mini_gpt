import torch
from torch.utils.data import DataLoader
from apex import amp
import os
from data_process.dataset_gpt3 import DatasetGPT3
from model.gpt3 import GPT3,GPT3Config
import torch.multiprocessing as MP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group

def ddp_setup(rank,world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        )



    
def prepare_dataloader(
            name:str,
            device_id:list[int],
            n_batch_size:int,
            vob_size:int,
            ):
        dataset = DatasetGPT3(
                sqlite_url='sqlite:///./data/'+name+'.db'
                )
        if_shuffle = True
        dataloader =None

        def collate_fn(
            tensor_list,
            ):

            max_length = max(tensor.size(0) for tensor in tensor_list)
            padded_tensors = [torch.cat([tensor, torch.full((max_length - tensor.size(0),), vob_size, dtype=tensor.dtype)]) 
                        for tensor in tensor_list]
            padded_tensor_batch = torch.stack(padded_tensors)
            
            return padded_tensor_batch

        if len(device_id) >1:
            if_shuffle = False
            dataloader = DataLoader(
                dataset, 
                batch_size=n_batch_size, 
                shuffle=if_shuffle,
                collate_fn=collate_fn,
                sampler=DistributedSampler(dataset)
                )
        else:
            dataloader = DataLoader(
                dataset, 
                batch_size=n_batch_size, 
                shuffle=if_shuffle,
                collate_fn=collate_fn
                )
        return dataloader

def get_train_objs(
        gpt3conf:GPT3Config,
        gpu_id:int,
    ):
    dataloader = prepare_dataloader(
        name=gpt3conf.data_set_name,
        device_id=gpt3conf.device_id,
        n_batch_size=gpt3conf.n_batch_size,
        vob_size=gpt3conf.vocab_size
    )
    if gpt3conf.if_gpu==1:
        device = torch.device(gpu_id)
    else:
        device = torch.device("cpu")
    gpt3 = GPT3(gpt3conf,device)
    if gpt3conf.pretrain_model == "":
        pass
    else:
        gpt3.load_state_dict(torch.load(gpt3conf.pretrain_model))
    gpt3.to(device)
    optimizer = torch.optim.AdamW(gpt3.parameters(), lr=gpt3conf.learning_rate)

    if gpt3conf.if_amp:
        gpt3, optimizer = amp.initialize(gpt3, optimizer, opt_level="O1")


    return dataloader,gpt3,optimizer

class trainer:
    def __init__(
            self,
            model:GPT3,
            optimizer:torch.optim.Optimizer,
            train_data:DataLoader,
            gpt3conf:GPT3Config,
            gpu_id:int
            ) -> None:
        self.gpt3conf = gpt3conf
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.device = torch.device(gpu_id)
        self.gpu_id = gpu_id

        if len(self.gpt3conf.device_id)>1:
            self.model = DDP(
                module=self.model,
                device_ids=[gpu_id],
                find_unused_parameters = True
                )
        

    

    def __run_batch(
            self,
            epoch:int,
            batch_idx:int,
            source,
            ):
        self.optimizer.zero_grad()
        loss = self.model(source)

        if self.gpt3conf.if_amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch {batch_idx}/{len(self.train_data)} | Loss {loss.item():.4f}")

        if batch_idx%self.gpt3conf.save_per_batchs ==0:

            if  self.gpu_id==self.gpt3conf.device_id[0]:
                torch.save(self.model.module.state_dict(), self.gpt3conf.output_path+'/model_iter_'+str(epoch)+'_batch_'+str(batch_idx)+'.pth')
            else:
                torch.save(self.model.state_dict(), self.gpt3conf.output_path+'/model_iter_'+str(epoch)+'_batch_'+str(batch_idx)+'.pth')
    

    def __run_epoch(self,epoch:int):
        b_sz  = len(self.train_data)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize {b_sz} | Steps: {len(self.train_data)}")

        for batch_idx, data in enumerate(self.train_data):
            data = data.to(self.device)
            self.__run_batch(epoch,batch_idx,data)

        if  self.gpu_id==self.gpt3conf.device_id[0]:
            torch.save(self.model.module.state_dict(), self.gpt3conf.output_path+'/model_iter_'+str(epoch)+'.pth')
        else:
            torch.save(self.model.state_dict(), self.gpt3conf.output_path+'/model_iter_'+str(epoch)+'.pth')
    
    def train(self):
         for epoch in range(0,self.gpt3conf.n_epoch):
             self.__run_epoch(epoch)

    

def train(rank:int,world_size:int,global_conf:GPT3Config):

    
    

    if len(global_conf.device_id)>1:
        ddp_setup(rank,world_size)



    if os.path.exists(global_conf.output_path):
        pass
    else:
        os.makedirs(global_conf.output_path)

    dataloader,gpt3,optimizer = get_train_objs(global_conf,rank)
    

    train_helper = trainer(
        model=gpt3,
        optimizer=optimizer,
        train_data = dataloader,
        gpt3conf=global_conf,
        gpu_id=rank
        )
    train_helper.train()
    destroy_process_group()


if __name__  == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
    global_conf = GPT3Config("./conf/gpt3_v3.yaml")
    #torch.cuda.set_device(global_conf.device_id)
    world_size = len(global_conf.device_id)
    MP.spawn(train,args=(world_size,global_conf,),nprocs=world_size)
                    
            