# internal phaze code
from .model import BaseModelIR
from ..utils import ShapeProp, PhazeGraph
from ..utils import custom_tracer_for_megatron
from ..utils import megatron_language_models

# torch module loads
import torch
from transformers import (BertTokenizer, BertModel,)
from transformers.utils import is_torch_fx_available

try:
    import pretrain_bert
    from megatron.training.initialize import initialize_megatron
    from megatron.core.enums import ModelType
    from megatron.training.global_vars import get_args
except:
    Warning("Megatron is not installed. Megatron models will not be supported for full import.")
    pass

import os
import sys
from pathlib import Path

if is_torch_fx_available():
    from transformers.utils.fx import (
        symbolic_trace as symbolic_trace_transformers,
    )


class BertIR(BaseModelIR):
    def __init__(self, model_name="bertbase", tmp_width=1, cp_degree=1, sp_enabled=False):
        super().__init__(model_name, tmp_width)

        self.out_dir = None
        self.graphmodule = None
        self.cp_degree = cp_degree
        self.sp_enabled = sp_enabled

        self.out_dir = self.create_out_dir()

    def set_model(self):
        self.trace_only_model = False

        if self.model_name == "bertbase":
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.configuration = self.model.config
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.model_name == "bertlarge":
            self.model = BertModel.from_pretrained("bert-large-uncased", attn_implementation="eager")
            self.configuration = self.model.config
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-large-uncased")
        elif self.model_name == "megatronbert":
            print(
                "MegatronBert is initialized with each tmp width. Current width ",
                self.tmp_width,
            )
            self.set_args_megatron()
            self.model = self.megatron_model_provider()
        else:
            raise TypeError("Model type not found in Bert", self.model_name)

    def megatron_model_provider(
        self,
        pre_process=True,
        post_process=True,
    ):
        """Build the megatron model."""

        if not torch.cuda.is_available():
            ValueError(
                "Cuda is not available on the machine, cannot extract megatron graph")

        device = torch.device("cuda")

        args = get_args()
        args.model_type = ModelType.encoder_or_decoder
        num_tokentypes = 2 if args.bert_binary_head else 0
        model = pretrain_bert.model_provider(
            pre_process=pre_process,
            post_process=post_process,
        ).to(device)

        model.half()

        return model

    def create_out_dir(self):
        curr_dir = Path(__file__).parent.absolute()
        curr_dir = os.path.join(curr_dir, "../out/Bert/")
        isExist = os.path.exists(curr_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(curr_dir)
            print("The new directory is created!")

        return curr_dir

    def get_model_type(self):
        return "Bert"

    def get_out_dir(self):
        if not self.out_dir:
            raise ValueError("Out directory not setup for", self.model_name)

        return self.out_dir
    
    def reset_megatron_globals(self):
        import megatron.core.num_microbatches_calculator as mb_calc
        for attr in ['_GLOBAL_NUM_MICROBATCHES_CALCULATOR', '_NUM_MICROBATCHES_CALCULATOR']:
            if hasattr(mb_calc, attr):
                setattr(mb_calc, attr, None)

        from megatron.core import rerun_state_machine
        if hasattr(rerun_state_machine, '_GLOBAL_RERUN_STATE_MACHINE'):
            rerun_state_machine._GLOBAL_RERUN_STATE_MACHINE = None

        from megatron.training import global_vars
        for attr in ['_GLOBAL_ARGS', '_GLOBAL_TOKENIZER', '_GLOBAL_TENSORBOARD_WRITER',
                    '_GLOBAL_ADLR_AUTORESUME', '_GLOBAL_TIMERS', '_GLOBAL_SIGNAL_HANDLER']:
            if hasattr(global_vars, attr):
                setattr(global_vars, attr, None)

    def set_args_megatron(self):
        WORLD_SIZE = self.tmp_width * self.cp_degree
        RANK = 0

        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = str(1)
        os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
        os.environ["RANK"] = str(RANK)
        os.environ["MASTER_PORT"] = "6000"
        os.environ["MASTER_ADDR"] = "localhost"

        TENSOR_MP_SIZE = self.tmp_width
        PIPELINE_MP_SIZE = 1
        SEQP_ENABLE = self.sp_enabled
        CP_SIZE = self.cp_degree


        DISTRIBUTED_ARGS = {
            "--nproc_per_node": WORLD_SIZE,
            "--nnodes": 1,
            "--node_rank": RANK,
            "--master_addr": "localhost",
            "--master_port": 6000,
            "--transformer-impl" : "local",
        }

        CHECKPOINT_PATH = self.out_dir
        VOCAB_FILE = "./nest/vocabfiles/bert-large-uncased-vocab.txt"
        DATA_PATH = "my-bert_text_sentence"

        BERT_ARGS = {  # This is Bert-Large, with sequence length of 512
            "--num-layers": 24,  # we will extend to 24 later 
            "--hidden-size": 1024,
            "--num-attention-heads": 16,
            "--seq-length": 512,
            "--max-position-embeddings": 512,
            "--lr": 0.0001,
            "--lr-decay-iters": 990000,
            "--train-iters": 2000000,
            "--min-lr": 0.00001,
            "--lr-warmup-fraction": 0.01,
            "--micro-batch-size": 1,
            "--global-batch-size": 1,
            "--vocab-file": VOCAB_FILE,
            "--split": "949, 50, 1",
            "--fp16": True,
            "--tokenizer-type": "BertWordPieceLowerCase",
            "--no-async-tensor-model-parallel-allreduce": True,
            # "--use-legacy-models": True,
            "--ckpt-format": "torch",
            "--spec": "local",
        }

        OUTPUT_ARGS = {
            "--log-interval": 10,
            "--save-interval": 500,
            "--eval-interval": 100,
            "--eval-iters": 10,
            "--activations-checkpoint-method": "uniform",
        }

        PARALLEL_ARGS = {
            "--tensor-model-parallel-size": TENSOR_MP_SIZE,
            "--pipeline-model-parallel-size": PIPELINE_MP_SIZE,
            "--DDP-impl": "torch",
            "--no-masked-softmax-fusion": True,
            "--context-parallel-size": CP_SIZE,
        }
        if SEQP_ENABLE:
            PARALLEL_ARGS["--sequence-parallel"] = True

        #ALL_ARGS = DISTRIBUTED_ARGS | BERT_ARGS | OUTPUT_ARGS | PARALLEL_ARGS
        ALL_ARGS = {**DISTRIBUTED_ARGS, **BERT_ARGS,
                    **OUTPUT_ARGS, **PARALLEL_ARGS}

        def add_args(argdict):
            for key, val in argdict.items():
                sys.argv.append(key)
                sys.argv.append(str(val))
        self.reset_megatron_globals()
        sys.argv = [sys.argv[0]] 
        add_args(ALL_ARGS)
        initialize_megatron(ignore_unknown_args=True)

    def print_graphmodule(self):
        self.graphmodule.print_readable()

    def add_more_layer_info(self, ex_num_layers, repeat_layer_id):
        print("\033[96m" + "Extending number of layers (only) in the graph to",
              ex_num_layers, "for model", self.model_name, "\033[0m")
        self.phazegraph.extend_layer_info_sans_graph(
            ex_num_layers, repeat_layer_id)

    def obtain_symbolic_trace_model(self, micro_batch_size=1, sequence_length=1):
        # extracting the graphmodule
        # megatron bert symbolic trace is specific for Megatron-LM
        # remainder models use transformer huggingface's symbolic trace function
        if self.model_name in megatron_language_models:
            device = torch.device("cuda")

            input_ids = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            attention_mask = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.float,).to(device)
            token_type_ids = torch.zeros(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            lm_labels = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.long,).to(device)
            
            inference_params = None
            packed_seq_params = None
            extra_block_kwargs = None
            runtime_gather_output = None

            graphmodule: torch.fx.GraphModule = custom_tracer_for_megatron(
                self.model)

            model_shapeprop = ShapeProp(graphmodule,self.tmp_width)
            self.graphmodule = model_shapeprop.propagate(
                input_ids, attention_mask, token_type_ids, lm_labels, inference_params, packed_seq_params, extra_block_kwargs, runtime_gather_output)

        else:
            input_ids = torch.ones(
                micro_batch_size, sequence_length, dtype=torch.int32)

            graphmodule: torch.fx.GraphModule = symbolic_trace_transformers(
                self.model)

            model_shapeprop = ShapeProp(graphmodule, self.tmp_width)
            self.graphmodule = model_shapeprop.propagate(input_ids)

    def get_layer_id(self, n, curr_layer_id):
        layer_annotations = ["layer", "layers"]

        node_name = n.name
        layer_details = node_name.split("_")
        for l in range(0, len(layer_details)):
            if layer_details[l] in layer_annotations:
                if layer_details[l + 1] and layer_details[l + 1].isdigit():
                    return (True, int(layer_details[l + 1]))

        return (False, 0)

    def create_graph_from_symbolic_trace(self):
        super().create_graph_from_symbolic_trace()

    def extract_model_graph(self, micro_batch_size=1, sequence_length=64, force_reextract_model=False):
        self.load_language_model(
            self.out_dir, micro_batch_size, sequence_length, force_reextract_model, sp_enabled=self.sp_enabled, cp_degree=self.cp_degree)

        # num_layers = 0
        # if self.model_name == "megatronbert" and micro_batch_size in [2, 4]:
        #     num_layers = 24

        # if num_layers :
        #     self.add_more_layer_info(num_layers, repeat_layer_id=0)