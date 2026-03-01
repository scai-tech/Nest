baseline_config = {
    
    "bertlarge": {
        "num_transformer_layers": 24,
        "p": 1,
        "d": 1,
        "t": 128,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 32,
        "d": 1,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 15,
        "d": 8,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 40,
        "d": 3,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 32,
        "d": 1,
        "t": 1,
        # "layer_partition": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]
    },
}
