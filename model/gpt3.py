import torch
import torch.nn as nn
from omegaconf import OmegaConf
import math

class GPT3Config():
    def __init__(self,config_file:str):
        self._load_config(config_file)
    
    def _load_config(self, config_file):
        config = OmegaConf.load(config_file)
        self.model_name = config["model_name"]
        self.n_attention_layer = config["n_attention_layer"]
        self.vocab_size = config["vocab_size"]
        self.n_hidden_size = config["n_hidden_size"]
        self.position_dim = config["position_dim"]
        self.n_positions = config["n_positions"]
        self.embed_dim = config["embed_dim"]
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
        

class SelfAttention(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        n_head:int,
        max_token:int,
        attn_pdrop:float,
        resid_pdrop:float,
        if_causal:bool,
        if_train:bool,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_head = n_head
        self.n_head_dim = self.input_dim // n_head
        self.if_causal = if_causal
        self.if_train = if_train
        assert self.n_head_dim * n_head == self.input_dim, "input_dim must be divisible by num_heads"
        
        self.qheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.kheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.vheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.c_proj = nn.Linear(self.input_dim,self.output_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_token, max_token), dtype=torch.bool)).view(
                1, 1, max_token, max_token
            ),
            persistent=False,
        )
        
    
    def forward(self,
        q:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor,
        attention_mask: torch.Tensor = None
        ):
        batch_, length_, _ = q.shape
        wq = self.qheads(q).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,1,3)
        wk = self.kheads(k).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,3,1)
        wv = self.vheads(v).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,1,3)
        attn_weights = torch.matmul(wq,wk)/ math.sqrt(self.n_head_dim) #(b,d,l,l)

        if self.if_causal:
            causal_mask = self.bias[:, :, 0:length_, 0:length_]
            #attn_weights = attn_weights + causal_mask[None, None, :, :]  # 
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  
            attention_mask = attention_mask.expand(-1, self.n_head, length_, -1)
            attn_weights = attn_weights + attention_mask

        att_score = nn.functional.softmax(attn_weights,dim=-1)
        attn_output = torch.matmul(att_score,wv).transpose(1,2).contiguous().reshape(batch_, length_, self.n_head*self.n_head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

class LayerNorm(nn.Module):
    def __init__(self,normalized_shape:int,eps=1e-5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(normalized_shape))
        self.b_2 = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self,
                tensor_in:torch.Tensor,
                )->torch.Tensor:
        mean = tensor_in.mean(dim=-1, keepdim=True)
        std = tensor_in.std(dim=-1, keepdim=True, unbiased=False)
        normalized_tensor = (tensor_in - mean) / (std + self.eps)
        return self.a_2 * normalized_tensor + self.b_2

class GPT2MLP(nn.Module):
    def __init__(self, gpt3conf:GPT3Config):
        super().__init__()
        self.c_fc = nn.Linear(gpt3conf.embed_dim, gpt3conf.n_hidden_size)
        self.c_proj = nn.Linear(gpt3conf.n_hidden_size, gpt3conf.embed_dim)
        self.act = torch.nn.GELU()
        self.dropout = nn.Dropout(gpt3conf.resid_pdrop)
        self.if_train = gpt3conf.if_train

    def forward(self, hidden_states:torch.Tensor ) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        if self.if_train:
            hidden_states = self.dropout(hidden_states)
        return hidden_states

class GPT3Block(nn.Module):
    def __init__(
        self,
        gpt3conf:GPT3Config,
        layer_idx:int,
        if_causal:bool,
        **kwargs):
        super().__init__(**kwargs)
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(gpt3conf.embed_dim)
        self.ln2 = nn.LayerNorm(gpt3conf.embed_dim)
        #"""
        self.att_layer = SelfAttention(
            input_dim=gpt3conf.embed_dim,
            output_dim=gpt3conf.embed_dim,
            n_head=gpt3conf.n_head,
            max_token = gpt3conf.max_token,
            attn_pdrop=gpt3conf.attn_pdrop,
            resid_pdrop=gpt3conf.resid_pdrop,
            if_causal=if_causal,
            if_train = gpt3conf.if_train,
            )
        #"""
        #self.att_layer = nn.MultiheadAttention(embed_dim=gpt3conf.embed_dim, num_heads=gpt3conf.n_head, batch_first=True)
        self.mlp = GPT2MLP(gpt3conf)
        
    def forward(self,x:torch.Tensor,mask):

        res_x = self.ln1(x)
        x =self.att_layer(res_x, res_x, res_x, mask)+x

        res_x = self.ln2(x)
        x_mlp = self.mlp(res_x)
        x = x + x_mlp
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
            gpt3conf.embed_dim,
            padding_idx=self.mask_idx
            )

    def forward(self,input_seq:torch.tensor):
        masked_embedded_seq = self.eb(input_seq)
        mask = torch.where(input_seq != self.mask_idx, torch.tensor(0), float('-inf'))
        #mask = (input_seq!=self.mask_idx)
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
        device,
        ):
        super().__init__()
        self.gpt3conf = gpt3conf
        self.device = device
        self.wte = MaskedEmbedding(self.gpt3conf)
        self.wpe = nn.Embedding(self.gpt3conf.n_positions,self.gpt3conf.position_dim)
        self.lpe = LearnedPositionalEmbedding(self.gpt3conf.max_token,self.gpt3conf.position_dim)
        self.decoder_layer = nn.ModuleList(
            [GPT3Block(gpt3conf=self.gpt3conf,layer_idx=i,if_causal=True) for i in range(0,self.gpt3conf.n_attention_layer)]
            )
        self.logits = nn.Linear(

            in_features=self.gpt3conf.embed_dim,
            out_features=self.gpt3conf.vocab_size+1
            
            )
        self.loss = nn.CrossEntropyLoss(ignore_index=gpt3conf.vocab_size) #20113

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

        embedding,mask = self.wte(label_tensor)
        
        position_embeddings = self.lpe(label_tensor)
        embedding = embedding+position_embeddings
        for module in self.decoder_layer:
            embedding = module(embedding,mask)
        logit = self.logits(embedding)
        logits_reshaped = logit[:,:-1,:].contiguous().view(-1, logit.shape[-1])  # Shape: (batch * length, vocab_size)
        labels_reshaped = label_tensor[:,1:].contiguous().view(-1).type(dtype=torch.LongTensor).to(self.device)
        

        return logits_reshaped,labels_reshaped





