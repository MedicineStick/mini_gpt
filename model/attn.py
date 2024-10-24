import torch
import torch.nn as nn
import math
from  torch.nn.functional import multi_head_attention_forward,scaled_dot_product_attention
# from xformers.ops import memory_efficient_attention
# https://github.com/pytorch/pytorch/issues/96099
        

def generate_causal_padding_mask(
            padding_mask: torch.Tensor,
            causal_mask: torch.Tensor,
            head_:int,
            length_:int,
            min_val:torch.Tensor,
            if_bool:bool,
            mask_dtype:torch.dtype
        )->torch.Tensor:
        # min_val : https://github.com/pytorch/pytorch/issues/103749
        padding_mask_expanded = padding_mask.unsqueeze(1)  # B,1,L
        attention_mask = padding_mask_expanded & padding_mask_expanded.transpose(1, 2)   # B,L,L
        attention_mask = attention_mask.unsqueeze(1).expand(-1, head_, -1, -1) # B,H,L,L
        causal_mask_sliced = causal_mask[:, :, 0:length_, 0:length_] # 1,1,L,L
        if if_bool:
            return attention_mask & causal_mask_sliced
        else:
            attention_mask = attention_mask & causal_mask_sliced
            attention_mask = torch.where(attention_mask==True,0.0,min_val)
            return attention_mask.to(mask_dtype)
        


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


class SelfAttention2(nn.Module):
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
        self.register_buffer(
            "bias",
            torch.ones(1,1,max_token, max_token, dtype=torch.bool).tril(diagonal=0),
            persistent=False,
        )
        self.qheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.kheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.vheads = nn.Linear(self.input_dim, self.n_head_dim * n_head)
        self.c_proj = nn.Linear(self.input_dim,self.output_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        
    
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

        
        if self.if_train:
            attention_mask = generate_causal_padding_mask(
                padding_mask=attention_mask,
                causal_mask=self.bias,
                head_=self.n_head,
                length_=length_,
                min_val=torch.finfo(attn_weights.dtype).min,
                if_bool=False,
                mask_dtype=attn_weights.dtype
                )
            expanded_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            # https://github.com/pytorch/pytorch/issues/103749
            #mask_value = torch.finfo(attn_weights.dtype).min
            #attn_weights = attn_weights.masked_fill(~expanded_mask, mask_value)
            attn_weights = attn_weights+expanded_mask


        att_score = nn.functional.softmax(attn_weights,dim=-1)
        attn_output = torch.matmul(att_score,wv).transpose(1,2).contiguous().reshape(batch_, length_, self.n_head*self.n_head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class SDPAttention(nn.Module):
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
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer(
            "bias",
            torch.ones(1,1,max_token, max_token, dtype=torch.bool).tril(diagonal=0),
            persistent=False,
        )
    
    
        
    
    def forward(self,
        q:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor,
        attention_mask: torch.Tensor = None,
        ):
        batch_, length_, _ = q.shape
        wq = self.qheads(q).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,1,3)
        wk = self.kheads(k).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,1,3)
        wv = self.vheads(v).view(batch_,length_,self.n_head,self.n_head_dim).permute(0,2,1,3)
        if self.if_train:
            attention_mask = generate_causal_padding_mask(
                padding_mask=attention_mask,
                causal_mask=self.bias,
                head_=self.n_head,
                length_=length_,
                min_val=torch.finfo(wq.dtype).min,
                if_bool=False,
                mask_dtype=wq.dtype
                )
        attn_output = scaled_dot_product_attention(
            query=wq,
            key=wk,
            value=wv,
            attn_mask=attention_mask,
            dropout_p=self.attn_pdrop,
            is_causal=False
            )
        #print("attn_output",attn_output.shape)
        attn_output = attn_output.reshape(batch_, length_, self.n_head*self.n_head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

