baseline_config = {
    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 8,
        "d": 32,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 32,
        "d": 2,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 8,
        "d": 32,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 80,
        "d": 3,
        "t": 1,
    },
    "mixtral": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 2,
        "t": 1,
    },


}
