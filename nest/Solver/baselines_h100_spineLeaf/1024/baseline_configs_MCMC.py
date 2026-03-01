baseline_config = {

    "megatronbert": {
        "num_transformer_layers": 24,
        "p": 1,
        "d": 1024,
        "t": 1,
    },
    
    "bertlarge": {
        "num_transformer_layers": 24,
        "p": 1,
        "d": 1024,
        "t": 1,
    },

    "megatrongpt3": {
        "num_transformer_layers": 96,
        "p": 33,
        "d": 7,
        "t": 4,
    },

    "llama2": {
        "num_transformer_layers": 32,
        "p": 15,
        "d": 68,
        "t": 1,
    },

    "llama3": {
        "num_transformer_layers": 80,
        "p": 40,
        "d": 25,
        "t": 1,
    },

    "mixtral": {
        "num_transformer_layers": 32,
        "p": 33,
        "d": 7,
        "t": 1,
        # "layer_partition": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]
    },
}
