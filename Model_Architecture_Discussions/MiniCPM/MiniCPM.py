import math
import warnings
from typing import List, Optional, Tuple, Union, Dict
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import re
from dataclasses import dataclass


import logging
from configuration_minicpm import MiniCPMConfig  # 直接导入

logger = logging.getLogger(__name__)


@dataclass
class BaseModelOutputWithPast(OrderedDict):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
@dataclass
class CausalLMOutputWithPast(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算了逆频率inv_freq并使用register_buffer方法将其注册为一个缓冲区
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 构建缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 计算并缓存余弦和正弦值
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # 将频率扩展到维度上
        emb = torch.cat((freqs, freqs), dim=-1)

        # 缓存余弦值和正弦值
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # 首先检查输入序列的长度是否超过了缓存的最大长度，如果超过了，则重新计算并缓存余弦和正弦值
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回对应序列长度的余弦和正弦值
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
        
def rotate_half(x):
    # 将输入张量 x 沿 emb 维度一分为二
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 将后半部分取负号，然后与前半部分拼接，对输入张量的隐藏维度进行旋转
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    # 保存原始数据类型
    orig_dtype = k.dtype  # torch.bfloat16
    
    # 根据 position_ids 选择 cos 和 sin，并在指定维度上扩展
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim] 便于和[bs, num_heads, q_len, head_dim] 维度的 q,k 进行矩阵乘法
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    
    # 将 q 和 k 转换为 float32 类型，以便进行精确的计算
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    
    # 计算 q 和 k 的旋转位置嵌入
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    
    # 将结果转换回原始数据类型并返回
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)  # [bs, num_heads, q_len, head_dim]


def create_causal_mask(input_shape, dtype, device, past_length=0):
    batch_size, query_length = input_shape
    # 创建一个上三角矩阵，填充最小浮点值，表示未来的token不能看到
    causal_mask = torch.triu(torch.full((query_length, query_length), torch.finfo(dtype).min, dtype=dtype, device=device), diagonal=1)
    # 如果有过去的key-value长度，则在mask前面添加零矩阵
    if past_length > 0:
        causal_mask = torch.cat([torch.zeros(query_length, past_length, dtype=dtype, device=device), causal_mask], dim=-1)
    # 扩展mask的维度以匹配批次大小，并返回
    return causal_mask[None, None, :, :].expand(batch_size, 1, query_length, query_length + past_length)

def expand_attention_mask(mask, dtype, target_length = None):
    batch_size, source_length = mask.shape
    target_length = target_length if target_length is not None else source_length

    # 扩展mask的维度以匹配目标长度和批次大小
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, target_length, source_length).to(dtype)
    # 反转mask，将1变为0，0变为1
    inverted_mask = 1.0 - expanded_mask
    # 将反转后的mask中为True的位置填充为最小浮点值
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

def prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: Union[torch.device, "str"] = "cpu",
):

    # 如果attention_mask存在且是2维的
    if attention_mask is not None and attention_mask.dim() == 2:
        # 获取批次大小和查询长度
        batch_size = attention_mask.shape[0]
        query_length = query_length
        # 更新input_shape和past_length
        input_shape = (batch_size, query_length)
        causal_mask = None
        if query_length > 1:
            # 创建4维的causal mask
            causal_mask = create_causal_mask(input_shape, dtype, device, past_length)
        # 扩展attention mask
        expanded_mask = expand_attention_mask(attention_mask, dtype, query_length)
        if causal_mask is not None:
            # 将causal mask中对应expanded mask为True的位置填充为最小浮点值
            expanded_attn_mask = causal_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)
        expanded_attn_mask = expanded_mask
    return expanded_attn_mask

class MiniCPMAttention(nn.Module):
    def __init__(self, config: MiniCPMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            layer_idx.warn_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout # 0.0
        self.hidden_size = config.hidden_size # 2304
        self.num_heads = config.num_attention_heads # 36
        self.head_dim = self.hidden_size // self.num_heads # 64
        self.num_key_value_heads = config.num_key_value_heads # 36
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 1
        self.max_position_embeddings = config.max_position_embeddings # 2048
        self.rope_theta = config.rope_theta  # 10000.0
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias) # (2304, 36*64=2304)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = MiniCPMRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value:  Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        # q,k,v 矩阵
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 拆成 num_heads 个头 (bsz, num_heads, q_len, self.head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None and len(past_key_value) > 0 and len(past_key_value[0]) > self.layer_idx and len(past_key_value[0][self.layer_idx].shape) > 1:
            # 如果有 kv-cache 缓存，需要加上缓存的长度
            kv_seq_len += past_key_value[0][self.layer_idx].shape[0] 
            
        # 获取 RoPE Embedding 对应位置的 cos 和 sin 值 （ 这里传入的 value_states 不会参与计算，只是确保类型和设备）
        cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)
        
        # 对 q 和 k 向量应用 RoPE 位置编码
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # 如果存在先前的 k-v 缓存
        if past_key_value is not None:
            # 若当前层缓存未初始化，则进行初始化
            if len(past_key_value[0]) <= self.layer_idx:
                # 为当前层新增 k-v 的缓存
                past_key_value[0].append(key_states)
                past_key_value[1].append(value_states)
            else:
                # 若当前层缓存已存在，通过在序列长度维度上进行拼接更新缓存
                past_key_value[0][self.layer_idx] = torch.cat([past_key_value[0][self.layer_idx], key_states], dim=-2)
                past_key_value[1][self.layer_idx] = torch.cat([past_key_value[1][self.layer_idx], value_states], dim=-2)

            key_states, value_states = past_key_value[0][self.layer_idx], past_key_value[1][self.layer_idx]   
            
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # 使用32位浮点数精度以提高计算精度
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
            
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 初始化权重参数为1，形状由hidden_size决定
        self.weight = nn.Parameter(torch.ones(hidden_size)) 
        # 设置方差的epsilon值，防止除以0
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 保存输入的数据类型，以便后续恢复
        old_dtype = hidden_states.dtype
        # 计算方差，先转换数据类型以提高精度，然后计算平方的均值
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        # 标准化隐藏状态，使用rsqrt（方差+epsilon的倒数根）进行缩放，并恢复原数据类型
        hidden_states = (hidden_states * torch.rsqrt(variance + self.variance_epsilon)).to(old_dtype)
        # 应用权重参数，进行缩放
        return hidden_states * self.weight
    
class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # 2304
        self.intermediate_size = config.intermediate_size # 5760
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x): 
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class MiniCPMPreTrainedModel(nn.Module):
    def __init__(self, *args, **kwargs):
        self.config = args[0]

        super().__init__()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiniCPMDecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniCPMAttention(config=config, layer_idx=layer_idx)

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        # 对输入归一化
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention 计算
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        # 应用残差连接并缩放
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        residual = hidden_states
        # 对 attention 结果归一化
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        # 应用残差连接并缩放
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    
class MiniCPMModel(MiniCPMPreTrainedModel):

    def __init__(self, config: MiniCPMConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # self._init_weights()
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        
        if use_cache:
            if past_key_values is not None and len(past_key_values) > 0 and len(past_key_values[0]) > 0 and len(past_key_values[0][0].shape) > 2:
                past_key_values_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        attention_mask = prepare_4d_causal_attention_mask(attention_mask, seq_length, past_key_values_length, inputs_embeds.dtype, inputs_embeds.device)
        
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # 对最终的结果归一化
        hidden_states = self.norm(hidden_states)

        # 添加最后一个解码器层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
            
class MiniCPMForCausalLM(MiniCPMPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取最后一层隐藏状态，并通过线性层（lm_head）转换为logits
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()
        
        loss = None
        # 如果存在标签，则进行损失计算
        if labels is not None:
            # 对logits和labels进行错位，以便预测下一个token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 为交叉熵损失计算准备，将tokens展平
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            # 计算交叉熵损失
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
        # 调整输入以匹配注意力掩码或过去的键值长度
        def adjust_input_ids(input_ids, attention_mask, past_length):
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                return input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                return input_ids[:, past_length:]
            return input_ids

        # 根据 kv 缓存的长度调整输入
        if past_key_values is not None and len(past_key_values) > 0 and len(past_key_values[0]) > 0 and len(past_key_values[0][0].shape) > 2:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

            input_ids = adjust_input_ids(input_ids, attention_mask, past_length)

            if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
                attention_mask = attention_mask[:, -max_cache_length:]
        
        # 按照注意力掩码生成位置ID
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
       
        # 更新模型输入
        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        else:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        
        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(history_str, return_tensors='pt').to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history
    
        '''进行推理'''
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=1024, temperature=1.0, top_k=None, use_cache=False, past_key_values=None, tokenizer=None, do_sample=False, **model_kwargs):
        if use_cache and past_key_values is None:
            # 初始化 kv 缓存
            past_key_values = ([], [])
            model_kwargs["past_key_values"] = past_key_values
        batch_size = input_ids.size(0)
        # 初始化完成标志和未完成序列标志
        finished = torch.zeros(batch_size, dtype=torch.bool).to(input_ids.device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool).to(input_ids.device)
        # 获取 pad_token_id 用于填充
        pad_token_id = tokenizer.pad_token_id  # 提前获取 pad_token_id

        for _ in range(max_new_tokens):
            # 准备生成的输入
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            logits = self(**model_inputs).logits[:, -1, :] / temperature  # Apply temperature
            
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
    
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)
            
            # 更新未完成序列的 next_tokens 
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences)        
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if "attention_mask" in model_kwargs:
                # 更新 attention_mask
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
            # 更新完成和未完成的序列标志
            finished |= (next_tokens.squeeze(-1) == tokenizer.eos_token_id)
            unfinished_sequences &= ~finished
            
            # 如果所有序列都完成，则停止生成
            if finished.all():
                break

        return input_ids