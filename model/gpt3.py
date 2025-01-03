import torch
import torch.nn as nn
from omegaconf import OmegaConf
import math
from model.attn import  SDPAttention,SelfAttention2

from transformers.loss.loss_utils import ForCausalLMLoss
 # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
from transformers.models.llama import modeling_llama


class GPT3Config():
    def __init__(self,config_file:str):
        self._load_config(config_file)
    
    def _load_config(self, config_file):
        config = OmegaConf.load(config_file)
        self.model_name = config["model_name"]
        self.n_attention_layer = config["n_attention_layer"]
        self.vocab_size = config["vocab_size"]
        self.n_hidden_size = config["n_hidden_size"]
        self.device_id = config["device_id"]
        self.n_batch_size = config["n_batch_size"]
        self.n_epoch = config["n_epoch"]
        self.n_head = config["n_head"]
        self.if_gpu = config["if_gpu"]
        self.learning_rate = config["learning_rate"]
        self.output_path = config["output_path"]
        self.resid_pdrop = config["resid_pdrop"]
        self.max_token = config["max_token"]
        self.max_gen_token = config["max_gen_token"]
        self.if_train = config["if_train"]
        self.save_per_batchs = config["save_per_batchs"]
        self.temperature = config["temperature"]
        self.top_k = config["top_k"]
        self.pretrain_model = config["pretrain_model"]
        self.wlist_size = config["wlist_size"]
        self.wlist = config["wlist"]
        self.attn_pdrop = config["attn_pdrop"]
        self.n_intermediate_size = config["n_intermediate_size"]
        self.mlp_bias = config["mlp_bias"]
        self.rms_norm_eps   = config["rms_norm_eps"]
        self.if_amp = bool(config["if_amp"])
        self.data_set_name = config["data_set_name"]
        self.log_file = self.output_path + '/train.log'
        self.clip_grad_norm = config["clip_grad_norm"]
        self.if_clip_grad = config["if_clip_grad"]
        self.weight_decay = config["weight_decay"]


class GPT3MLP(nn.Module):
    def __init__(self, gpt3conf:GPT3Config):
        super().__init__()

        self.hidden_size = gpt3conf.n_hidden_size
        self.n_intermediate_size = gpt3conf.n_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.n_intermediate_size, bias=gpt3conf.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.n_intermediate_size, bias=gpt3conf.mlp_bias)
        self.down_proj = nn.Linear(self.n_intermediate_size, self.hidden_size, bias=gpt3conf.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x:torch.Tensor ) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class GPT3Block(nn.Module):
    def __init__(
        self,
        gpt3conf:GPT3Config,
        layer_idx:int,
        if_causal:bool,
        **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx
        self.ln1 = nn.RMSNorm(gpt3conf.n_hidden_size)
        self.ln2 = nn.RMSNorm(gpt3conf.n_hidden_size)
        #"""
        self.att_layer = SDPAttention(
            input_dim=gpt3conf.n_hidden_size,
            output_dim=gpt3conf.n_hidden_size,
            n_head=gpt3conf.n_head,
            max_token = gpt3conf.max_token,
            attn_pdrop=gpt3conf.attn_pdrop,
            resid_pdrop=gpt3conf.resid_pdrop,
            if_causal=if_causal,
            if_train = gpt3conf.if_train,
            )
        #"""
        #self.att_layer = nn.MultiheadAttention(n_hidden_size=gpt3conf.n_hidden_size, num_heads=gpt3conf.n_head, batch_first=True)
        self.mlp = GPT3MLP(gpt3conf)
        
    def forward(self,x:torch.Tensor,mask):

        residual = x
        res_x = self.ln1(x)
        x =self.att_layer(res_x, res_x, res_x, mask)

        x = residual + x
        residual = x

        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_size):
        """
        Initialize the learned positional embedding module.
        
        Parameters:
        - max_seq_len (int): The maximum length of the sequences.
        - embed_size (int): The size of each embedding vector.
        """
        super(LearnedPositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, embed_size)

    def forward(self, x):
        """
        Forward pass for generating positional embeddings.
        
        Parameters:
        - x (torch.Tensor): A tensor of shape (batch_size, seq_len).
        
        Returns:
        - torch.Tensor: A tensor of shape (batch_size, seq_len, embed_size) containing
          positional embeddings for each position in the sequence.
        """
        # Generate a sequence of positions
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        
        # Retrieve the positional embeddings
        pos_embeddings = self.position_embeddings(positions)
        
        return pos_embeddings

class MaskedEmbedding(nn.Module):
    def __init__(
        self,
        gpt3conf:GPT3Config,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.mask_idx = gpt3conf.vocab_size
        self.eb = nn.Embedding(
            gpt3conf.vocab_size+1,
            gpt3conf.n_hidden_size,
            padding_idx=self.mask_idx
            )

    def forward(self,input_seq:torch.tensor):
        masked_embedded_seq = self.eb(input_seq)
        mask = torch.where(input_seq != self.mask_idx, True, False)
        return masked_embedded_seq,mask

class DynamicPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(DynamicPositionalEncoding, self).__init__()
        self.d_model = d_model
        # Precompute div_term for efficiency

    def forward(self, x):
        seq_length = x.size(1)
        position = torch.arange(seq_length, dtype=x.dtype, device=x.device).unsqueeze(1)
        positional_encoding = torch.zeros((seq_length, self.d_model), device=x.device, dtype=x.dtype)
        div_term = torch.exp(torch.arange(0.0, self.d_model, 2.0) * -(math.log(10000.0) / self.d_model)).to(x.device).type(x.dtype)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        positional_encoding = positional_encoding.unsqueeze(0).expand_as(x)
        return  positional_encoding


class GPT3(nn.Module):
    def __init__(
        self,
        gpt3conf:GPT3Config,
        gpu_id,
        ):
        super().__init__()
        if gpt3conf.if_gpu==1:
            self.device = torch.device(gpu_id)
        else:
            self.device = torch.device("cpu")
        print(f"initialize GPT3 model on gpu {gpu_id}...")
        self.gpt3conf = gpt3conf
        self.wte = MaskedEmbedding(self.gpt3conf)
        self.wpe = nn.Embedding(
            num_embeddings=self.gpt3conf.vocab_size+1,
            embedding_dim=self.gpt3conf.n_hidden_size,
            padding_idx=self.gpt3conf.vocab_size
            )
        self.lpe = LearnedPositionalEmbedding(self.gpt3conf.max_token,self.gpt3conf.n_hidden_size)
        self.decoder_layer = nn.ModuleList(
            [GPT3Block(gpt3conf=self.gpt3conf,layer_idx=i,if_causal=True) for i in range(0,self.gpt3conf.n_attention_layer)]
            )
        self.norm = nn.RMSNorm(self.gpt3conf.n_hidden_size)
        self.logits = nn.Linear(

            in_features=self.gpt3conf.n_hidden_size,
            out_features=self.gpt3conf.vocab_size+1
            
            )
        #self.loss = nn.CrossEntropyLoss(ignore_index=gpt3conf.vocab_size) #20113

        self.apply(self._init_weights)

    def set_test(self):
        for module in self.decoder_layer:
            module.if_causal = False

    @torch.no_grad()
    def inference(self,
                  prompt_tensor,
                  eos_id:int = 1):
        _,l = prompt_tensor.shape
        for i in range(0,self.gpt3conf.max_gen_token):
            embedding,_ = self.wte(prompt_tensor)
            
            position_embeddings = self.lpe(prompt_tensor)
            embedding = embedding+position_embeddings
            for module in self.decoder_layer:
                embedding = module(embedding,None)
            embedding = self.norm(embedding)
            logit = self.logits(embedding[:,-1:,:])
            logit = logit /  float(self.gpt3conf.temperature)
            probs = torch.nn.functional.softmax(logit, dim=-1)
            # Apply top-k sampling
            top_probs, top_indices = torch.topk(probs, self.gpt3conf.top_k, dim=-1)
            top_probs = top_probs / torch.sum(top_probs, dim=-1, keepdim=True)  # Re-normalize the probabilities
            
            # Sample from the top k probabilities
            sampled_indices = torch.multinomial(top_probs.squeeze(), num_samples=1)
            sampled_value = top_indices.squeeze(0)[:,sampled_indices]
            
            if sampled_value.item() == eos_id:
                return prompt_tensor

            prompt_tensor = torch.cat((prompt_tensor, sampled_value), dim=1)

        return prompt_tensor


    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,label_tensor):
        
        _,l = label_tensor.shape
        if l>self.gpt3conf.max_token:
            label_tensor = label_tensor[:,0:self.gpt3conf.max_token]
        else:
            pass
        #with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
        embedding,mask = self.wte(label_tensor)
        
        position_embeddings = self.lpe(label_tensor)
        embedding = embedding+position_embeddings
        for module in self.decoder_layer:
            embedding = module(embedding,mask)
        embedding = self.norm(embedding)
        logit = self.logits(embedding)
        loss = ForCausalLMLoss(
            logits=logit,
            labels=label_tensor,
            vocab_size=self.gpt3conf.vocab_size+1,
            ignore_index=self.gpt3conf.vocab_size,
        )
        """
        logits_reshaped = logit[:,:-1,:].contiguous().view(-1, logit.shape[-1])  # Shape: (batch * length, vocab_size)
        labels_reshaped = label_tensor[:,1:].contiguous().view(-1).type(dtype=torch.LongTensor).to(self.device)
        loss = self.loss(logits_reshaped, labels_reshaped)
        """
        return loss





