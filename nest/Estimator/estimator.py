# internal phaze imports
from nest.GraphExtractor.utils import generate_out_filename

from .architecture import set_configs_to_explore
from .architecture import tc_configs, vc_configs, bandwidth, bytes_per_element, frequency, only_explore_specific_configs

from .utils import convert_phaze_to_fused_graph, initialize_accelerator, reset_accelerator, get_engine_type
from .utils import initialize_network, reset_network
from .utils import tensor_core_estimator, vector_core_estimator, allreduce_estimator, alltoall_estimator, gather_estimator, scatter_estimator
from .utils import phaze_coretype_mapping, e_tuple

import configs

# python imports
import os
import json
import traceback
from pathlib import Path

estimates_dir = ""
global_estimates_filepath = ""
existing_estimates = {}


def setup_architecure():
    # Setting the architecture
    reset_accelerator()
    initialize_accelerator()
    set_configs_to_explore()

def setup_network():
    # Setting the Network
    reset_network()
    initialize_network()


def populate_estimates(tmpc_models, max_tmp_width, micro_batch_size, sequence_length, ep_degree, sp_enabled, cp_degree):
    global estimates_dir, global_estimates_filepath, existing_estimates
    # append configs.setup_name to the estimates directory to be estimates_<setup_name>
    estimates_dir = os.path.join(Path(__file__).parent.absolute(), f"estimates_{configs.setup_name}")
    # Ensure the directory exists
    os.makedirs(estimates_dir, exist_ok=True)

    global_estimates_filepath = os.path.join(estimates_dir, "global_estimates.json")

    # Initialize the file if it does not exist
    if not os.path.exists(global_estimates_filepath):
        with open(global_estimates_filepath, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    # Load existing estimates
    with open(global_estimates_filepath, "r", encoding="utf-8") as f:
        existing_estimates = json.load(f)

    setup_architecure()
    setup_network()

    latency_estimates = {}

    model_type = tmpc_models[0].get_model_type()
    model_name = tmpc_models[0].model_name

    model_estimates_dir = os.path.join(estimates_dir, model_type)
    estimate_file_name = generate_out_filename(
        model_name, "json", micro_batch_size, max_tmp_width, sequence_length,ep_degree=ep_degree, sp_enabled=sp_enabled, cp_degree=cp_degree)

    if only_explore_specific_configs:
        for model in tmpc_models:
            latency_estimates[str(model.tmp_width)
                              ] = append_latency_estimates(model)

    else:
        estimate_filepath = os.path.join(
            model_estimates_dir, estimate_file_name)

        if not os.path.isfile(estimate_filepath):
            if not os.path.exists(model_estimates_dir):
                os.makedirs(model_estimates_dir)

            for model in tmpc_models:
                latency_estimates[str(model.tmp_width)
                                  ] = append_latency_estimates(model)

            with open(estimate_filepath, "w") as f:
                try:
                    json.dump(latency_estimates, f,
                              ensure_ascii=False, indent=4)
                except:
                    os.remove(estimate_filepath)
                    print('\33[93m' + "Error writing to file: ",
                          estimate_filepath + '\033[0m')
            f.close

        else:
            latency_estimates = json.load(open(estimate_filepath))

    return latency_estimates


def write_global_estimates():
    with open(global_estimates_filepath, "w+") as f:
        json.dump(existing_estimates, f, ensure_ascii=False, indent=2)
    f.close()


def append_latency_estimates(model):
    # generate operator graphs
    # graphs with fused operators, to ensure operators across layers are not fused
    graphs = model.get_unique_op_graphs()
    phaze_graph = model.get_phaze_graph().get_graph()
    fusedgraphs = [convert_phaze_to_fused_graph(graph) for graph in graphs]

    global existing_estimates

    # estimates dictionary
    tc_estimates, vc_estimates, ar_estimates = {}, {}, {}

    def rd_wr_estimates_global(core_config, op_dim, e=None,):
        core_key = str(core_config)
        dim_key = "dim" + str(op_dim)

        ret_e = None
        if core_key in existing_estimates.keys():
            if dim_key in existing_estimates[core_key].keys():
                ret_e = e_tuple(*existing_estimates[core_key][dim_key])

        elif ret_e is None and e is not None:
            if core_key not in existing_estimates.keys():
                existing_estimates[core_key] = {}
            existing_estimates[core_key][dim_key] = e

        return ret_e

    try:
        # # for profiling use 
        # fwd_set = False
        # bwd_set = False
        # # Third party imports
        # from collections import namedtuple
        # e_tuple = namedtuple("estimate", ["latency", "energy", "utilization", "estimation_time"])
        # efwd = ebwd = e_tuple(0, 0, 0, 0,)
        # empty = {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}


        for fusedgraph in fusedgraphs:
            print("Estimating for layer_id", list(
                fusedgraph.nodes.values())[0].layer_id)
            for node in list(fusedgraph.nodes.values()):
                core_type = phaze_coretype_mapping[get_engine_type(
                    node.node_desc)]

                if (node.node_desc == "AllReduceBwd" or node.node_desc == "AllReduceFwd" or node.node_desc == "AllReduce"):
                    e = allreduce_estimator(
                        phaze_graph.nodes[node.node_id], model.tmp_width, node.node_desc, bandwidth, bytes_per_element,)
                    ar_estimates[str(node.node_id)] = e
                elif (node.node_desc == "AllToAll"):
                    print("AllToAll with ep_degree", model.ep_degree)
                    e = alltoall_estimator(
                        phaze_graph.nodes[node.node_id], model.ep_degree, node.node_desc, bandwidth, bytes_per_element,)
                    ar_estimates[str(node.node_id)] = e
                elif (node.node_desc == "AllGather"):
                    num_device = model.tmp_width # all gather could be for context parallel or Sequence parallel (with TMP)
                    if ("all_gather_context_parallel" in phaze_graph.nodes[node.node_id]["node"].get_name().__name__):
                        num_device = model.cp_degree
                    print("All gather with devices: ", num_device)
                    e = gather_estimator(
                        phaze_graph.nodes[node.node_id], num_device, node.node_desc, bandwidth, bytes_per_element,)
                    ar_estimates[str(node.node_id)] = e
                elif (node.node_desc == "ReduceScatter"):
                    print("Reduce Scatter with tmp degree", model.tmp_width)
                    e = scatter_estimator(
                        phaze_graph.nodes[node.node_id], model.tmp_width, node.node_desc, bandwidth, bytes_per_element,)
                    ar_estimates[str(node.node_id)] = e

                elif (core_type == "TC" or core_type == "TCandVC"):
                    tc_estimates[str(node.node_id)] = []
                    for cc in tc_configs:
                        # if fwd_set !=True:
                        #     efwd = e_tuple(0.0138, 0, 0, 0,)
                        #     ebwd = e_tuple(0.02179, 0, 0, 0,)
                        #     e = {"fwd": efwd._asdict(), "bwd": ebwd._asdict()}
                        #     tc_estimates[str(node.node_id)].append(e)
                        #     fwd_set = True
                        # else:
                        #     tc_estimates[str(node.node_id)].append(empty)
                        e = tensor_core_estimator(
                            fusedgraph, node, cc, rd_wr_estimates_global, frequency)
                        tc_estimates[str(node.node_id)].append(e)

                elif (core_type == "VC"):
                    vc_estimates[str(node.node_id)] = []
                    for cc in vc_configs:
                        e = vector_core_estimator(
                            fusedgraph, node, cc, frequency)
                        vc_estimates[str(node.node_id)].append(e)
                        # vc_estimates[str(node.node_id)].append(empty)

        write_global_estimates()
    except:
        write_global_estimates()
        traceback.print_exc()
        exit("Estimates error out!")

    return {"TC": tc_estimates, "VC": vc_estimates, "AR": ar_estimates, }
