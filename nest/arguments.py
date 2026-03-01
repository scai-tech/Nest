# python imports
import argparse


def process_arguments():
    parser = argparse.ArgumentParser()

    ### Execution mode
    parser.add_argument("--model_names", type=str, nargs="*", required=True,
                        default=None, help="A list of model names from the supported group of models",)
    parser.add_argument("--force_reextract_model", action="store_true", required=False,
                        default=False, help="Force the model extractor to reload the model",)
    parser.add_argument("--exec_type", type=str, required=False, default="run_solver",
                        help="extract_graph - extracts torch.fx graphs, prepopulate_estimates, run_solver runs the algorthmic solver\
                        \\ creates a library of runtime estimates", choices=["run_solver", "prepopulate_estimates", "extract_graph"],)
    parser.add_argument("--use_ilp", action="store_true", required=False,
                        default=False, help="Use ILP for operator Scheduling (False, uses sequential scheduling)",)
    ### Runtime options
    parser.add_argument("--sequence_length", type=int, required=False, default=64,
                        help="sequence length for language models. Bert uses 512 and GPT uses 2048 sequence \
                        \\ length. Default is 64 independent of the model",)
    parser.add_argument("--micro_batch_size", nargs='+', type=int, required=False,
                        default=1, help="micro batch sizes to extract graph",)
    parser.add_argument("--activation_recomputation", action="store_true", required=False, default=False, 
                        help="Enable Activation Recomputation",)
    parser.add_argument("--hbm_size", nargs='+', type=int, required=False, default=32,
                        help="HBM sizes to explore in GB \
                        \\ Default is 32GB.",)
    ### Parallelization details 
    parser.add_argument("--max_tmp_width", type=int, required=False, default=1,
                        help="Maximum tensor model parallel width. MegatronBert and MegatronGPT is 8 as per literature. \
                        \\ Other models do not support this option and should use the default. Default is 1.",)
    parser.add_argument("--exp_parallel_degree", type=int, required=False, default=1,
                        help="expert parallel decgee for MoE models. Default is 1.",)
    parser.add_argument("--enable_seq_parallel", action="store_true", required=False, default=False,
                        help="Enable Sequence Parallel (NOTE: Requires TMP for MegatronLM style split)",)
    parser.add_argument("--context_parallel_degree", type=int, required=False, default=1,
                        help="context parallel decgee for models. Default is 1.",)
    ### Network Setup
    parser.add_argument("--num_accelerators", type=int, default=1024, 
                        help="Number of Accelerators in the cluster",)
    parser.add_argument("--bandwidths_per_level", nargs='+', type=float, required=False,
                        default=None, help="bandwidth of network at each level, default set at /Estimator/configs",)
    parser.add_argument("--devices_per_level", nargs='+', type=int, required=False,
                        default=None, help="devices of network at each level, default set at /Estimator/configs",)
    parser.add_argument("--latency_per_level", nargs='+', type=float, required=False,
                        default=None, help="latency of network at each level, default set at /Estimator/configs",)
    ### Network Configuration Name
    parser.add_argument("--setup_name", type=str, required=False, default="default",
                         help="Name of the network and accelerator configuration, estimates, solver outputs and baseline configs will be stored under this name",)
    args = parser.parse_args()

    return args
