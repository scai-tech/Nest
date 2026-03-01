baseline_config = {
    
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 3,
        "d": 21,
        "t": 2,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 8,
        "d": 2,
        "t": 8,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 4,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 80,
        "d": 1,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 1,
        "t": 1,
    },
}
