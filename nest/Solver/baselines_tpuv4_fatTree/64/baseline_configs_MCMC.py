baseline_config = {
    
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 5,
        "d": 6,
        "t": 2,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 16,
        "d": 1,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 2,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 41,
        "d": 1,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 16,
        "d": 1,
        "t": 1,
    },
}
