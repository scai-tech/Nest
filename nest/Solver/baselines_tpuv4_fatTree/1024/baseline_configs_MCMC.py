baseline_config = {
    
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 5,
        "d": 204,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 9,
        "d": 14,
        "t": 8,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 9,
        "d": 98,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 80,
        "d": 12,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 8,
        "t": 1,
    },
}
