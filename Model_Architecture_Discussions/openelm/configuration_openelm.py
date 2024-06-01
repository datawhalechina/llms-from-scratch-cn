"""Implements HF OpenELMConfig based on PretrainedConfig"""
from numbers import Number
from typing import List, Optional, Union

import numpy as np
from transformers import PretrainedConfig


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62

    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute the number of heads.

    Args:
        model_dim: Model dimension.
        head_dim: Head dimension.

    Returns:
        An integer denoting number of heads in multi-head attention is returned.

    Raises:
        ValueError: if model dimension is not divisible by head dimension.
    """
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}."
        )


OpenELM_CONFIGS = {
    "OpenELM-270M": dict(
        num_transformer_layers=16,
        model_dim=1280,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-450M": dict(
        num_transformer_layers=20,
        model_dim=1536,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-1_1B": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-3B": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multipliers to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
}


class OpenELMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OpenELMModel`]. It is used to instantiate an OpenELM model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the OpenELM model.
        max_context_length (`int`, *optional*, defaults to 2048):
            Maximum number of input tokens.
        num_transformer_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        model_dim (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        qkv_multipliers (`Union[Number, List[Number]]`, *optional*, defaults to 1.0):
            If the qkv_multipliers is a Number, then all attention layers have the same latent dimensions,
            resulting in uniform allocation of parameters.
            If the qkv_multipliers is a List of Number, then each attention layer have different latent dimensions
            assuming qkv_multipliers[0] != qkv_multipliers[1]. This results in variable allocation of parameters in attention layer.
            This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        num_query_heads (`Union[int, None]`, *optional*, defaults to None):
            The number of query heads, computed from `compute_heads(model_dim=model_dim, head_dim=head_dim)`.
        num_gqa_groups (`int`, *optional*, defaults to 1):
            This variable allows to switch between multi-head attention, group query attention, and multi-query attention.
            When num_gqa_groups == 1, then it is multi-head attention.
            When 1 < num_gqa_groups < num_heads and num_heads is divisible by num_gqa_groups, then it is group query attention
            When num_gqa_groups == num_heads, then it is multi-query attention
        ffn_multipliers (`Union[Number, List[Number]]`, *optional*, defaults to 4.0):
            Feed-forward network (FFN) multipliers.
            If the ffn_multipliers is a Number, then all FFN layers have the same latent dimensions,
            resulting in uniform allocation of parameters.
            If the ffn_multipliers is a List of Number, then each FFN layer have different latent dimensions
            assuming ffn_multipliers[0] != ffn_multipliers[1]. This results in variable allocation of parameters in FFN layer.
            This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
        ffn_with_glu (`bool`, *optional*, defaults to True):
            Whether to use FFN with Gated Linear Unit (GLU)
        ffn_dim_divisor (`int`, *optional*, defaults to 256):
            The ffn layer dimension divisor.
        activation_fn_name (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the decoder.
        normalization_layer_name (`str` or `function`, *optional*, defaults to `"rms_norm"`):
            Type of normalization layer.
        normalize_qk_projections (`bool`, *optional*, defaults to False):
            Whether to normalize queries and keys after projections
        share_input_output_layers (`bool`, *optional*, defaults to False):
            Whether to share the embedding between input and output linear layer
        rope_freq_constant (`int`, *optional*, defaults to 10000):
            The base period of the RoPE embeddings.
        rope_max_length (`int`, *optional*, defaults to 4096):
            That rope_max_length is set to twice of max_context_length.
            This allows flexibility in token lengths during training or fine-tuning.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
    """

    model_type = "openelm"

    def __init__(
        self,
        vocab_size: int = 32000,
        max_context_length: int = 2048,
        num_transformer_layers: int = 12,
        model_dim: int = 2048,
        head_dim: int = 128,
        qkv_multipliers: Union[Number, List[Number]] = 1.0,
        num_query_heads: Union[int, None] = None,
        num_gqa_groups: int = 1,
        ffn_multipliers: Union[Number, List[Number]] = 4.0,
        ffn_with_glu: bool = True,
        ffn_dim_divisor: int = 256,
        activation_fn_name: str = "swish",
        normalization_layer_name: str = "rms_norm",
        normalize_qk_projections: bool = False,
        share_input_output_layers: bool = False,
        rope_freq_constant: int = 10000,
        rope_max_length: int = 4096,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_context_length = max_context_length
        self.num_transformer_layers = num_transformer_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.qkv_multipliers = qkv_multipliers
        self.num_query_heads = num_query_heads
        self.num_gqa_groups = num_gqa_groups
        self.ffn_multipliers = ffn_multipliers
        self.ffn_with_glu = ffn_with_glu
        self.ffn_dim_divisor = ffn_dim_divisor
        self.activation_fn_name = activation_fn_name
        self.normalization_layer_name = normalization_layer_name
        self.normalize_qk_projections = normalize_qk_projections
        self.share_input_output_layers = share_input_output_layers
        self.rope_freq_constant = rope_freq_constant
        self.rope_max_length = rope_max_length
        self.num_query_heads = (
            compute_heads(model_dim=model_dim, head_dim=head_dim)
            if num_query_heads is None
            else num_query_heads
        )
        self.initializer_range = initializer_range

        self.__post_init__()
        super().__init__(
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def __post_init__(self) -> None:
        if self.num_gqa_groups is not None:
            head_multiple_of = self.num_gqa_groups
        else:
            head_multiple_of = 2

        if isinstance(self.qkv_multipliers, Number):
            # All attention layers have the same latent dimensions, resulting in uniform allocation of parameters.
            qkv_dim = make_divisible(
                self.model_dim * self.qkv_multipliers,
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif (
            isinstance(self.qkv_multipliers, (tuple, list))
            and len(self.qkv_multipliers) == 2
        ):
            # Each attention layer have different latent dimensions assuming qkv_multipliers[0] != qkv_multipliers[1].
            # This results in variable allocation of parameters in attention layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            qkv_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            # Make sure that scaled model dimension is divisible by scaled head dimension.
            query_dims = [
                int(
                    make_divisible(
                        self.model_dim * m, divisor=self.head_dim * head_multiple_of
                    )
                )
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # compute the number of query, key, and value heads
        # For multi-head and multi-query attention, the number of heads for query, key, and value are the same.
        # For group query attention, the number of key and value heads are the same.
        self.num_query_heads = [
            int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims
        ]
        self.num_kv_heads = [
            q_heads // self.num_gqa_groups for q_heads in self.num_query_heads
        ]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif isinstance(self.ffn_multipliers, (tuple, list)):
            # Each FFN layer have different latent dimensions assuming ffn_multipliers[0] != ffn_multipliers[1].
            # This results in variable allocation of parameters in FFN layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            if len(self.ffn_multipliers) == 2:
                self.ffn_multipliers = [
                    round(v, 2)
                    for v in np.linspace(
                        self.ffn_multipliers[0],
                        self.ffn_multipliers[1],
                        num=self.num_transformer_layers,
                        dtype=float,
                    )
                ]
            else:
                assert (
                    len(self.ffn_multipliers) == self.num_transformer_layers
                ), f"{len(self.ffn_multipliers)=}!={self.num_transformer_layers=}"
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # check num_query_heads divisible by num_kv_heads for every layer
        for layer_idx in range(len(query_dims)):
            assert self.num_query_heads[layer_idx] % self.num_kv_heads[layer_idx] == 0
