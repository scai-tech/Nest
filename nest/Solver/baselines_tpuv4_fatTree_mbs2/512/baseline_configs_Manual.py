baseline_config = {
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 8,
        "d": 64,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 32,
        "d": 4,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 8,
        "d": 64,
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
