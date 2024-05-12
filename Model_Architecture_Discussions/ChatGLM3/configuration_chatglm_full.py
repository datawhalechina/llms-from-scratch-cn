import torch

class ChatGLMConfig():
    model_type = "chatglm"
    # original_rope = 'https://123.com'
    attention_softmax_in_fp32 = True
    max_length = 8196

    multi_query_attention = True
    multi_query_group_num = 2
    tie_word_embeddings = False
    num_layers = 28
    padded_vocab_size = 65024
    hidden_size = 4096
    ffn_hidden_size = 13696
    kv_channels = 128
    num_attention_heads = 32
    seq_length = 2048
    hidden_dropout = 0.0
    classifier_dropout = None
    attention_dropout = 0.0
    layernorm_epsilon = 1e-5
    rmsnorm = True
    apply_residual_connection_post_layernorm = False
    post_layer_norm = True
    add_bias_linear = False
    add_qkv_bias = True
    bias_dropout_fusion = True
    apply_query_key_layer_scaling = True
    attention_softmax_in_fp32 = True
    fp32_residual_connection = False
    quantization_bit = 0
    pre_seq_len = None
    prefix_projection = False

    torch_dtype = torch.float16
    vocab_size = 65024
    pad_token_id = 0
    eos_token_id = 2

    use_cache = False
    use_return_dict = True
    output_hidden_states = False