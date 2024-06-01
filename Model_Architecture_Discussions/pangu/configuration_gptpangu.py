from transformers.configuration_utils import PretrainedConfig


class GPTPanguConfig(PretrainedConfig):
    model_type = "gpt_pangu"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=40000,
        max_position_embeddings=1024,
        hidden_size=1024,
        intermediate_size=None,
        num_layers=24,
        num_heads=16,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        use_cache=True,
        # bos_token_id=9,
        # eos_token_id=9,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_weights = scale_attn_weights
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.use_cache = use_cache

        # self.bos_token_id = bos_token_id
        # self.eos_token_id = eos_token_id

        super().__init__(**kwargs)
