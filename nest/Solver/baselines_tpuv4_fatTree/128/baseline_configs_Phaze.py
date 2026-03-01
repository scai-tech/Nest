baseline_config = {


    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 1,
        "d": 128,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 16,
        "d": 2,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 6,
        "d": 21,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 41,
        "d": 3,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 16,
        "d": 2,
        "t": 1,
    },
}
