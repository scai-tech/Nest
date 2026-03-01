baseline_config = {


    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 1,
        "d": 1024,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 16,
        "d": 16,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 6,
        "d": 164,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 41,
        "d": 23,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 16,
        "d": 16,
        "t": 1,
    },
}
