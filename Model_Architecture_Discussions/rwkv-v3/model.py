########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np
import math, json, time, types, copy, sys, os
import torch
from torch.nn import functional as F
import torch.nn as nn

from transformers import PreTrainedTokenizerFast

RUN_DEVICE = 'cpu' # cpu cuda
ctx_len = 768
n_layer = 12
n_embd = 768
# n_layer = 24
# n_embd = 1024

# ---> download RWKV-3 169M model from https://huggingface.co/BlinkDL/rwkv-3-pile-169m/tree/main

# MODEL_NAME = '/data1/ckw/RWKV-3-Pile-430M-20220817-10602'
MODEL_NAME = '/data1/ckw/RWKV-3-Pile-20220720-10704'
K_EPS = 1e-8

vocab_size = 50277
VOCAB_NAME = '20B_tokenizer.json'

print(f'\n* running on {RUN_DEVICE}')

################################################################################################################

class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd))
        
        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(n_embd, 1))
        self.time_curve = torch.tensor([-(ctx_len - 2 - i) for i in range(ctx_len-1)]).unsqueeze(0)
        self.time_first = nn.Parameter(torch.ones(n_embd, 1) * math.log(0.3))
        
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_k = nn.Parameter(torch.ones(1,1,n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1,1,n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1,1,n_embd))

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)

        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        k = torch.clamp(k, max=60)
        k = torch.exp(k)

        kv = k * v

        self.time_w = torch.cat([torch.exp(self.time_decay) * self.time_curve.to(self.time_decay.device), self.time_first], dim=-1)
        w = torch.exp(self.time_w)
        
        w = w[:,-T:].unsqueeze(1)
        wkv = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(kv), w, groups=C)
        wk = F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w, groups=C) + K_EPS

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        
        rwkv = self.output(rwkv)
        return rwkv

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        
        self.att = RWKV_TimeMix(layer_id)
        self.ffn = RWKV_ChannelMix(layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class RWKV_GPT(nn.Module):
    def __init__(self, MODEL_NAME=MODEL_NAME):
        super().__init__()
        print('\nloading RWKV-GPT', MODEL_NAME)

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)
        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))
        self.eval()

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)

        return x

################################################################################################################

time_buf = {}

class RWKV_RNN():
    def __init__(self, MODEL_NAME=MODEL_NAME):
        print('\nloading RWKV-RNN', MODEL_NAME)
        self.ctx_len = ctx_len
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=VOCAB_NAME)

        self.w = types.SimpleNamespace()
        
        w = torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))

        for x in w.keys():
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x]))
            if '.time_first' in x:
                w[x] = torch.exp(w[x])
                    
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.clear()
    
    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)

    def LN(self, xx, w):
        return F.layer_norm(xx, (n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)
        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)

        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)
        k = torch.square(torch.relu(w.key.weight @ xk))
        kv = w.value.weight @ k

        return r * kv

    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(n_embd, device=RUN_DEVICE)
            self.aa[name] = torch.zeros(n_embd, device=RUN_DEVICE)
            self.bb[name] = torch.zeros(n_embd, device=RUN_DEVICE)

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)

        self.xx[name] = xx

        r = torch.sigmoid(w.receptance.weight @ xr)

        k = torch.exp(torch.clamp(w.key.weight @ xk, max=60))
        v = w.value.weight @ xv
        kv = k * v

        a = self.aa[name] + w.time_first * kv
        b = self.bb[name] + w.time_first * k
        self.aa[name] = w.time_decay * self.aa[name] + kv
        self.bb[name] = w.time_decay * self.bb[name] + k

        rwkv = r * a / (b + K_EPS)

        return w.output.weight @ rwkv

    def run(self, ctx):
        w = self.w
        x = w.emb.weight[ctx[-1]]

        x = self.LN(x, w.blocks[0].ln0) #相比v2版本，增加了一个初始的归一化
        for i in range(n_layer):
            x = x + self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')
            x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')

        x = self.LN(x, w.ln_out)

        x = w.head.weight @ x
        x = x.tolist()

        return x

################################################################################################################