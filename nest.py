# internal imports
from nest.arguments import process_arguments
from nest.exec_modes import extract_and_prepopulate, extract_and_solve, extract_only
from nest.GraphExtractor import supported_models

import time
from nest.Estimator import configs


def main(args):

    mbs = args.micro_batch_size
    model_names = args.model_names
    exec_type = args.exec_type
    seq_len = args.sequence_length
    max_tmpc = args.max_tmp_width
    force_reextract_model = args.force_reextract_model
    hbm_size_list = args.hbm_size
    ep_degree = args.exp_parallel_degree
    sp_enabled = args.enable_seq_parallel
    cp_degree = args.context_parallel_degree
    ar = args.activation_recomputation

    configs.set_config(args)

    assert set(model_names).issubset(
        set(supported_models)
    ), "Model not supported by Phaze. Please check the list of supported models in GraphExtractor.py"

    print("Running for hbm: ", hbm_size_list, " mbs: ",
          mbs, " max tmp: ", max_tmpc, " sp enabled: ", sp_enabled)
    if (ep_degree > 1):
        print("ep degree:", ep_degree)
    if (cp_degree > 1):
        print("cp degree:", cp_degree)
    if exec_type == "extract_graph":
        # Extract graph using Torch.fx
        # Fill details about the tensor sizes - weights, activations, and
        # intermediate results
        for micro_batch_size in mbs:
            extract_only(model_names, max_tmpc, micro_batch_size,
                         seq_len, force_reextract_model,ep_degree, sp_enabled, cp_degree)

    elif exec_type == "prepopulate_estimates":
        # Every node has a corresponding estimates in a 3D matrix <TMP
        # strategy, core dimensions, and number of cores>
        for micro_batch_size in mbs:
            extract_and_prepopulate(model_names, max_tmpc,
                                    micro_batch_size, seq_len, force_reextract_model,ep_degree, sp_enabled, cp_degree)

    elif exec_type == "run_solver":

        # initialize variables for final "best" config
        final_config = None
        final_total_time = 0
        final_ilp_time = 0
        final_dp_time = 0
        final_estimation_time = 0
        final_micro_batch_size = 0
        final_hbm_size = 0
        final_throughput = 0
        final_activation_recomputation = False

        # search both activation recomp true and false
        activation_recomputations = [ar]

        for micro_batch_size in mbs:
            for hbm_size in hbm_size_list:
                for activation_recomputation in activation_recomputations:

                    print("mbs: ", micro_batch_size, " HBM size: " + str(hbm_size) +
                          " activation_recomputation: " + str(activation_recomputation))

                    start = time.time()

                    final_config, estimation_time, ilp_time, dp_time = extract_and_solve(
                        model_names, max_tmpc, micro_batch_size, seq_len, force_reextract_model, activation_recomputation, hbm_size*1024*1024*1024, ep_degree, sp_enabled, cp_degree)

                    end = time.time()

                    print("Best phaze config for mbs: ", micro_batch_size, " HBM: ", hbm_size, " Activation Recomp: ", activation_recomputation, "\n",
                          "Config ", final_config, "\n",
                          "Models", model_names, "\n",
                          "total solving time, ilptime and dptime", end - start, ilp_time, dp_time, "\n",
                          "estimation time", estimation_time)

                    final_total_time += end - start
                    final_ilp_time += ilp_time
                    final_dp_time += dp_time

                    cc, strategy = final_config

                    if strategy != None:
                        if strategy[model_names[0]].throughput > final_throughput:
                            final_config = final_config
                            final_micro_batch_size = micro_batch_size
                            final_hbm_size = hbm_size
                            final_throughput = strategy[model_names[0]].throughput
                            final_activation_recomputation = activation_recomputation
            final_estimation_time += estimation_time

        if final_config != None:
            print("Best config for single model comparison: mbs: ", final_micro_batch_size, " HBM: ", final_hbm_size, " Activation Recomp: ", final_activation_recomputation, "\n",
                  "Config ", final_config, "\n",
                  "Model", strategy[model_names[0]], "\n",
                  "total solving time, ilptime and dptime", final_total_time, final_ilp_time, final_dp_time, "\n",
                  "estimation time", final_estimation_time)
        else:
            print("No valid configuration found")


if __name__ == "__main__":
    args = process_arguments()
    main(args)
