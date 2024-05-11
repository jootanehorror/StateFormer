import torch
from torch import nn
import torch.nn.functional as F
from torch import einsum, nn
from einops import rearrange, repeat, pack, unpack
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from .bimamba import BiMamba
from .biasnorm import BiasNorm
from mamba_ssm.ops.triton.layernorm import RMSNorm
from mamba_ssm import Mamba
import math

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)




class TSiLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha_scale = nn.Parameter(torch.zeros((dim)))
        self.beta_scale = nn.Parameter(torch.zeros((dim)))
        self.bias = nn.Parameter(torch.zeros((dim)))
   
    
    def forward(self, x):
        alpha = self.alpha_scale.exp()
        
        beta = self.beta_scale.exp()

        gate = F.sigmoid(x * beta + self.bias)
       
        x =  alpha * x * gate
        
        return x
    

class CGLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.m = math.sqrt(2)



  def forward(self,x):

    gate = torch.special.erfc( x**2 /4 - 1/self.m )

    
    x = x * gate * F.tanh(F.softplus(x))
    return x
  


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size ,stride,padding=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(dim_in,dim_out ,kernel_size=kernel_size,stride=stride, padding=padding,bias= False)
   
    
        self.norm= BiasNorm(dim_out,-2)
        
        
    def forward(self, x):
        x = F.mish(self.conv_in(x))
       

        x = self.norm(x)
        
        return x
    



class GatedConvBlock(nn.Module):
    def __init__(self, d_model,kernel_size=5, ff_mult=1):
        super().__init__()

        self.padding = calc_same_padding(kernel_size)

        hidden_dim = int(d_model * ff_mult)
        
        self.ff = nn.Linear(d_model, 3 * hidden_dim, bias = False)

        self.log_scale = nn.Parameter(torch.zeros((hidden_dim)))

        self.ff_out = nn.Linear(hidden_dim,d_model, bias = False)

        self.act =TSiLU(hidden_dim)

        self.l_act = CGLU()
        
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, groups=hidden_dim, bias= False)
        
        self.norm = BiasNorm(hidden_dim,-2)
        
        
    def forward(self, x):
        
        x, y ,gate = self.ff(x).chunk(3, dim = -1)

        x = self.l_act(x)

        gate = gate * self.act(y)

        gate = F.pad(self.norm(gate.transpose(2,1)),self.padding)
        
        gate = F.dropout(self.dwconv(gate).transpose(2,1), p=0.05 , training=self.training)

        x = self.ff_out(x * gate * self.log_scale.exp())
    
        return x
    


class FastMultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head, kv_head=None,causal=False,window_size=(-1,-1)):
        super().__init__()
        self.causal = causal
        self.n_head = n_head
        if kv_head is None:
            self.kv_nhead = n_head
            kv_state = n_state
        else:
            self.kv_nhead =kv_head
            kv_state = n_state//n_head * kv_head
        self.q = nn.Linear(n_state, n_state, bias = False)
        
        self.window_size=window_size
        self.kv = nn.Linear(n_state, 2 * kv_state, bias=False)
        self.out = nn.Linear(n_state, n_state,  bias=False)
        
        self.scale = 1. / math.sqrt(n_state//n_head)




    def forward(
            self,
            x,
            xa,
    ):
        h = self.n_head * x.size(0)
        kvh = self.kv_nhead * x.size(0)

        q = self.q(x)
        k,v = self.kv(xa).chunk(2, dim = -1)
        



        q = rearrange(q, 'b n (h d) -> b n h d', h = h)
        k = rearrange(k, 'b n (h d) -> b n h d', h = kvh)
        v = rearrange(v, 'b n (h d) -> b n h d', h = kvh)


        out = flash_attn_func(q, k, v, softmax_scale=self.scale,causal=self.causal,window_size=self.window_size,dropout_p=0.15)

        

        return self.out(out.flatten(start_dim=2))



class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device

        t = torch.arange(n, device = device).type_as(self.inv_freq)

        sinu = torch.einsum('i , j -> i j', t, self.inv_freq)

        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)

        return emb * self.scale



    


class StateFormerBlock(nn.Module):
    def __init__(self,d_model):
        super().__init__()

        
        self.fcb = GatedConvBlock(d_model,kernel_size=15,ff_mult=2)
        
        self.fcb_norm = RMSNorm(d_model)
            
        self.mixer = BiMamba(d_model)

        self.norm = RMSNorm(d_model)




    def forward(self, x):
        
        x = x + self.fcb(self.fcb_norm(x))

        x = x + self.mixer(self.norm(x))

        return x
    


class DecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 heads,
                 kv_heads=None
                 ):
        super().__init__()


        
        self.mixer_norm = RMSNorm(d_model)
        
        self.mixer = Mamba(d_model=d_model)

        self.attn_norm = RMSNorm(d_model)

        self.cross_attn = FastMultiHeadAttention(d_model, heads,kv_head=kv_heads)
        
    

    def forward(self, x, xa):
        
        x = x + self.mixer(self.mixer_norm(x))
        
        x = x + self.cross_attn(self.attn_norm(x), xa)

        return x 
    



class StateFormer(nn.Module):
    def __init__(
            self,
            n_mels,
            d_model,
            n_layer,
            

            
 
    ):
        super(StateFormer, self).__init__()

        
        self.conv1 = ConvBlock(n_mels, d_model,  kernel_size=1, stride = 1,padding=0)

        self.conv2 = ConvBlock(d_model, d_model, kernel_size=3, stride = 2)

        self.embed = ScaledSinuEmbedding(d_model)
        
        self.blocks  = nn.ModuleList([StateFormerBlock(d_model) for _ in range(n_layer)])

        self.post_norm = RMSNorm(d_model)
        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)


        x = x.transpose(2,1)

        x = x + self.embed(x)

        for block in self.blocks:
            x = block(x)

        x= self.post_norm(x)
            
        return x
    


class Decoder(nn.Module):
    def __init__(
            self,
            n_vocab,
            d_model,
            heads,
            n_layer,
            padding_idx = 50257        
    ):
       
        super(Decoder, self).__init__()
        
        self.embed_tokens = nn.Embedding(n_vocab, d_model, padding_idx=padding_idx)
        
        
        self.blocks = nn.ModuleList([DecoderBlock(d_model=d_model, heads =heads) for k_ in range(n_layer)])

        self.norm = RMSNorm(d_model)

        self.mixer = Mamba(d_model, d_conv=2, expand=4)
    
        self.post_norm = RMSNorm(d_model)


    def forward(self, tokens, xa):

        x = self.embed_tokens(tokens)

        for block in self.blocks:
            x = block(x, xa)
  
        x = x + F.dropout(self.mixer(self.norm(x)), p=0.2, training=self.training)
        x = self.post_norm(x)

        x = F.linear(x, self.embed_tokens.weight.to(x.dtype)) 

        return x
    



class StateFormerSeq2Seq(nn.Module):
    def __init__(self,config):
        super().__init__(config)

        self.encoder = StateFormer(n_mels=config.n_mels,d_model=config.d_model,n_layer=config.encoder_n_layer)
        self.decoder = Decoder(n_vocab=config.n_vocab,d_model=config.d_model,heads=config.heads,n_layer=config.decoder_n_layer)
        

        self.pad_token = self.decoder.padding_idx
        self.start_token = self.decoder.start_idx
        self.vocab_size = self.decoder.vocab_size

    def forward(
            self,
            input_features = None,
            decoder_input_ids = None):

        
        x  = self.encoder(input_features)

        lm_logits= self.decoder(decoder_input_ids, x)  
            
        return lm_logits
