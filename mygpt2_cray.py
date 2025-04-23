import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   # 确保嵌入维度能被头数整除
        # key, query, value projections for all heads, but in a batch
        # 合并QKV投影（核心组件1）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # 输入n_embd，输出3*n_embd
        # output projection
        # 输出投影（核心组件2）        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)   # 保持维度不变
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        # 注册因果掩码,下三角矩阵
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        # 生成QKV三元组（关键步骤）
        qkv = self.c_attn(x)    # [B,T,3*C]
        # 分割为独立Q/K/V（为什么这么赋值？detail in report）
        q, k, v = qkv.split(self.n_embd, dim=2) # 每个都是[B,T,C]
        # 多头维度重组
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention (materializes the large (T,T) matrix for all the queries and keys)
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 调用PyTorch 提供的底层高效内核,替换上面的4行语句
        # ​​高效自注意力（Self-Attention）实现​​，专门优化了 Transformer 类模型的核心计算
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        # 最终输出投影,恢复原始维度
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
#@dataclass 是 Python 中的一个装饰器（decorator），用于自动为类生成特殊方法（如 __init__、__repr__ 等），从而简化类的定义。
@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    '''
    每个 token 被映射到一个 768 维的向量空间,768 = 12(头数) × 64(每个头的维度)
    GPT-2 不同版本中的维度：
    GPT-2 (小型): 768 维
    GPT-2 Medium: 1024 维
    GPT-2 Large: 1280 维
    GPT-2 XL: 1600 维
    '''
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #def forward(self, idx):
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        # 生成位置索引 [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        # 位置嵌入，将位置转换为向量
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        # 词嵌入，将token ID转换为向量
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        # 依次通过每个 Transformer 块
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        # 最后的层归一化
        x = self.transformer.ln_f(x)
        # 通过语言模型头部转换为词表大小的logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        # return logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # @classmethod 装饰器用于声明一个类方法：
    # 类方法可以直接通过类名调用，而不需要先创建类的实例
    # 类方法的第一个参数是 cls（代表类本身），而不是 self（代表实例）

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        #（1）设置模型参数
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        print("inint sd.keys()")
        # print(sd.keys())
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # （2）加载huggingface已训练好模型及状态字典
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # （3）将参数复制到GPT
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------------------

# 使用类方法创建实例
import time
# attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
     device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
     device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=4, T=32)

#train_loader = DataLoaderLite(B=16, T=1024)
torch.set_float32_matmul_precision('high')
'''
# get a data batch
import tiktoken
enc = tiktoken.get_encoding('gpt2')
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1])
buf = buf.to(device)

x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)
'''

# 从gpt2加载已训练好的参数
# model = GPT.from_pretrained('gpt2')
# model.to('cuda')

# get logits
#model = GPT(GPTConfig())
model = GPT(GPTConfig(vocab_size=50304))
# ​​模型编译优化接口​​，用于显著提升模型训练和推理速度（通常可加速 10%-30%）
model = torch.compile(model)
model.to(device)
# logits, loss = model(x, y)
# print(loss)

# optimize!
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    #logits, loss = model(x, y)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # print(f"step {i}, loss: {loss.item()}")
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    #dt = (t1 - t0)*1000 # time difference in miliseconds
    #tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    #print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f"step {i:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; 
sys.exit(0)

# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
print(tokens)
# x = tokens.to('cuda')
x = tokens.to(device)
# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        # 只取最后一个位置的预测 [:, -1, :]
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        print(topk_probs)
        print(topk_indices)

        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        # 使用 top-k 采样而不总是选择概率最高的token,增加文本多样性
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        print(ix)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        print(xcol)
        # 5个生成序列的第一个token分别如下，index值实际上也是token值
        # not，and，and，I，a
        for i in range(num_return_sequences):
            print(enc.decode(xcol[i].tolist()))
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)
        
    break
# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)