"""
Mamba在一个PyTorch文件中的简单、极简实现。

建议在阅读代码之前或期间阅读以下内容：
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu 和 Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush 和 Sidd Karamcheti)
        https://srush.github.io/annotated-s4

术语表：
    b: 批次大小                       (`B` 在 Mamba 论文 [1] 算法2中)
    l: 序列长度                      (`L` 在 [1] 算法2中)
    d 或 d_model: 隐藏维度
    n 或 d_state: 潜在状态维度      (`N` 在 [1] 算法2中)
    expand: 扩展因子                (`E` 在 [1] 第3.4节中)
    d_in 或 d_inner: d * expand     (`D` 在 [1] 算法2中)
    A, B, C, D: 状态空间参数        (参见任何状态空间表示公式)
                                        (B, C是输入相关的(即选择性的，这是Mamba的一个关键创新); A, D不是)
    Δ 或 delta: 输入相关的步长
    dt_rank: Δ的秩                  (参见[1] 第3.6节 "Parameterization of ∆")

"""
from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """完整的Mamba模型。"""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # 将输出投影与嵌入权重绑定。
                                                     # 参见 "Weight Tying" 论文


    def forward(self, input_ids):
        """
        参数:
            input_ids (long tensor): 形状 (b, l)    (参见顶部术语表中b, l, d_in, n的定义)
    
        返回:
            logits: 形状 (b, l, vocab_size)

        官方实现:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """从HuggingFace加载预训练权重到模型中。
    
        参数:
            pretrained_model_name: 以下之一
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        返回:
            model: 加载了权重的Mamba模型
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """包装Mamba块的简单块，具有归一化和残差连接。"""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        参数:
            x: 形状 (b, l, d)    (参见顶部术语表中b, l, d_in, n的定义)
    
        返回:
            output: 形状 (b, l, d)

        官方实现:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            注意: 官方库链式残差块看起来像
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            其中第一个Add是无操作。这纯粹是为了性能原因，因为这
            允许他们融合Add->Norm。

            我们相反实现的块更为熟悉，更简单，数值上等效
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """一个单一的Mamba块，如Mamba论文[1]第3.4节中的图3所述。"""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj接收`x`并输出输入特定的Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj将Δ从dt_rank投影到d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba块前向。这与Mamba论文[1]第3.4节中的图3相同。
    
        参数:
            x: 形状 (b, l, d)    (参见顶部术语表中b, l, d_in, n的定义)
    
        返回:
            output: 形状 (b, l, d)
        
        官方实现:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # 形状 (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """运行SSM。参见：
            - Mamba论文[1]第3.2节中的算法2
            - The Annotated S4 [2] 中的run_SSM(A, B, C, u)

        参数:
            x: 形状 (b, l, d_in)    (参见顶部术语表中b, l, d_in, n的定义)
    
        返回:
            output: 形状 (b, l, d_in)

        官方实现:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # 计算 ∆ A B C D，状态空间参数。
        #     A, D 与输入无关 (参见Mamba论文[1]第3.5.2节"Interpretation of A"中为何A不是选择性的)
        #     ∆, B, C 与输入相关 (这是Mamba与线性时不变S4的一个关键区别，
        #                          也是Mamba称为**选择性**状态空间的原因)
        
        A = -torch.exp(self.A_log.float())  # 形状 (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # 这类似于The Annotated S4 [2]中的run_SSM(A, B, C, u)
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """执行选择性扫描算法。参见：
            - Mamba论文[1]中的第2节状态空间模型
            - Mamba论文[1]第3.2节中的算法2
            - The Annotated S4 [2]中的run_SSM(A, B, C, u)

        这是经典的离散状态空间公式：
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        除了B和C(以及用于离散化的步长delta)依赖于输入x(t)。
    
        参数:
            u: 形状 (b, l, d_in)    (参见顶部术语表中b, l, d_in, n的定义)
            delta: 形状 (b, l, d_in)
            A: 形状 (d_in, n)
            B: 形状 (b, l, n)
            C: 形状 (b, l, n)
            D: 形状 (d_in,)
    
        返回:
            output: 形状 (b, l, d_in)
    
        官方实现:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            注意: 我将`selective_scan_ref`中的一些部分进行了重构，所以功能不完全匹配。
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # 离散化连续参数 (A, B)
        # - A 使用零阶保持(ZOH)离散化 (参见Mamba论文[1]第2节方程4)
        # - B 使用简化的欧拉离散化而不是ZOH。从与作者的讨论中：
        #   "A是更重要的项，简化B对性能影响不大"
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # 执行选择性扫描 (参见The Annotated S4 [2]中的scan_SSM())
        # 注意，以下是顺序的，而官方实现是一个更快的并行扫描，此外还考虑了硬件 (如FlashAttention)。
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # 形状 (b, l, d_in)
        
        y = y + u * D
    
        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
