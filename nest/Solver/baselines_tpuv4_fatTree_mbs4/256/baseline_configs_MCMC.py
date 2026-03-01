baseline_config = {
    
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 8,
        "d": 32,
        "t": 1,
    },
    "llama2": {
        "num_transformer_layers": 32,
        "p": 30,
        "d": 8,
        "t": 1,
        "layer_partition": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1]
    },
}
