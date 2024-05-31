### RWKV v1 源码详细解读

由于时间久远，没有找到对应的预训练脚本，对于v1主要做源码分析，v2-v6均可以找到载入模型实现的脚本。本文档详细解析 RWKV v1 版本的核心代码，具体包括初始化、时间混合模块、通道混合模块和多头注意力机制。代码使用 PyTorch 实现，具有良好的可读性和扩展性。

---

#### 模型初始化

首先，我们定义了一个用于初始化线性层和嵌入层的函数 `RWKV_Init`。

```python
def RWKV_Init(module, config): 
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters(): 
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0  
            scale = 1.0 

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: 
                    scale = config.rwkv_emb_scale

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd: 
                    scale = config.rwkv_emb_scale

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0, std=-gain)
```

该函数遍历模块中的所有线性层和嵌入层，并根据特定条件初始化其权重。对于线性层，若偏置存在，则将其初始化为零；对于嵌入层，计算权重的增益和缩放因子。根据不同的条件使用不同的初始化方法，例如正交初始化或正态初始化。

---

#### 时间混合模块

`RWKV_TimeMix` 类实现了时间混合机制。

```python
class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        with torch.no_grad(): 
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)]) 
            for h in range(config.n_head):
                if h < config.n_head - 1:
                    decay_speed = math.pow(config.ctx_len, -(h+1)/(config.n_head-1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
                
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)

        self.output = nn.Linear(config.n_attn, config.n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT-1:] 
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60) 
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)

        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k

        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]
```

该类实现了时间混合机制，通过时间权重矩阵 `time_w` 对输入进行变换。`time_w` 矩阵根据头数和上下文长度初始化，随后对输入进行时间维度上的变换和混合。`key`、`value` 和 `receptance` 三个线性层分别生成键、值和接收信号，并通过 sigmoid 函数计算输出。

---

#### 通道混合模块

`RWKV_ChannelMix` 类实现了通道混合机制。

```python
class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        
        hidden_sz = 5 * config.n_ffn // 2 
        self.key = nn.Linear(config.n_embd, hidden_sz)
        self.value = nn.Linear(config.n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, config.n_embd)
        self.receptance = nn.Linear(config.n_embd, config.n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        
        x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        
        wkv = self.weight(F.mish(k) * v) 

        rwkv = torch.sigmoid(r) * wkv

        return rwkv
```

该类实现了通道混合机制，通过 `key`、`value` 和 `receptance` 三个线性层对输入进行变换。`key` 和 `value` 层生成的张量经过 `mish` 激活函数变换后再通过 `weight` 层进行加权变换，最后与 `receptance` 层生成的接收信号相乘得到最终输出。

---

#### 多头注意力机制

`MHA_rotary` 类实现了多头注意力机制，并引入了旋转位置编码。

```python
class MHA_rotary(nn.Module):
    def __init__(self, config, layer_id, time_shift = False):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        if time_shift:
            self.time_shift = nn.ZeroPad2d((0,0,1,-1))

        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)

        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))
        
        self.rotary_ndims = int(self.head_size * 0.5)
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)

        self.output = nn.Linear(config.n_attn, config.n_embd)

    def forward(self, x):
        B, T, C = x.size()

        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C//2]), x[:, :, C//2:]], dim = -1)

        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2

)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)

        x = att @ v
        x = x.transpose(1, 2).contiguous().view(B, T, -1)

        x = self.output(x)
        return x
```

该类实现了多头注意力机制，并引入了旋转位置编码以增强模型的位置信息表示。通过 `query`、`key` 和 `value` 三个线性层生成查询、键和值，再通过旋转编码和自注意力机制计算最终输出。

---

#### 总结

RWKV v1 模型实现了基础的时间混合、通道混合和多头注意力机制。通过对输入进行多维度的变换和混合，实现了对复杂特征的提取和表示。上述代码段展示了模型的核心组件和主要计算过程，为后续版本的优化和改进提供了坚实的基础。