import logging
import json

logger = logging.getLogger(__name__)

class MiniCPMConfig():

    model_type = "minicpm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        scale_emb=1,
        dim_model_base=1,
        scale_depth=1,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        use_return_dict=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.scale_emb = scale_emb
        self.dim_model_base = dim_model_base
        self.scale_depth = scale_depth
        self.pad_token_id=pad_token_id
        self.bos_token_id=bos_token_id
        self.eos_token_id=eos_token_id
        self.tie_word_embeddings=tie_word_embeddings
        self.output_attentions=output_attentions
        self.output_hidden_states=output_hidden_states
        self.return_dict=return_dict
        self.use_return_dict=use_return_dict
        
    def to_json_string(self) -> str:
        config_dict = self.__dict__
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"
