baseline_config = {
    
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 2,
        "d": 256,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 9,
        "d": 7,
        "t": 8,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 11,
        "d": 46,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 80,
        "d": 6,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 4,
        "t": 1,
    },
}
