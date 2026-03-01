# internal phaze imports
# from GraphExtractor.utils import generate_out_filename

# from .architecture import set_configs_to_explore
# from .architecture import tc_configs, vc_configs, bandwidth, bytes_per_element, frequency, only_explore_specific_configs

# from .utils import convert_phaze_to_fused_graph, initialize_accelerator, reset_accelerator, get_engine_type
# from .utils import initialize_network, reset_network
# from .utils import tensor_core_estimator, vector_core_estimator, allreduce_estimator, alltoall_estimator, gather_estimator
from alpa.estimator.estimator_wrapper import phaze_coretype_mapping, e_tuple
from alpa.estimator.wham.op_to_compute import get_compute_unit
from alpa.estimator.estimator_wrapper import tensor_core_estimator, vector_core_estimator, allreduce_estimator, alltoall_estimator, allgather_estimator, wham_op_rep
from collections import namedtuple

def get_engine_type(node_desc):
    return get_compute_unit(node_desc)

# python imports
import os
import json
import traceback
from pathlib import Path
FREQUENCY = 1.05 * (10**6) 

per_core_config = namedtuple(
    "per_core_config", ["num", "width", "depth", "GLB_Buffer"])

HBM = 64

VC_config = per_core_config(
    num=2,
    width=128,
    depth=1,
    GLB_Buffer=134217728  # 128 MB
)
TC_config = per_core_config(
    num=8,
    width=128,
    depth=128,
    GLB_Buffer=134217728 # 128 MB
)

estimate_cache = {}

def create_key( op_info):
    """
    Creates a unique, hashable key from an operation dictionary.
    It converts the lists of inputs and outputs into immutable tuples.
    """
    op_name = op_info["op_name"]
    # Convert lists to tuples to make them hashable
    inputs_tuple = tuple(op_info["inputs"])
    outputs_tuple = tuple(op_info["outputs"])
    
    return (op_name, inputs_tuple, outputs_tuple)

def get_cached_result(op_info):
        """
        Checks if an operation's result is in the cache.
        
        Args:
            op_info (dict): The dictionary describing the operation.

        Returns:
            The cached result if found, otherwise None.
        """
        key = create_key(op_info)
        return estimate_cache.get(key, None)

def cache_result(op_info, result):
    """
    Adds an operation's result to the cache.
    """
    key = create_key(op_info)
    estimate_cache[key] = result
            



# estimates_dir = os.path.join(Path(__file__).parent.absolute(), "estimates/")
# global_estimates_filepath = os.path.join(estimates_dir, "global_estimates.json")

# exisiting_estimates = {}

# if not os.path.exists(global_estimates_filepath):
#     with open(global_estimates_filepath, "w+") as f:
#         json.dump({}, f, ensure_ascii=False, indent=2)
#     f.close()

# existing_estimates = json.load(open(global_estimates_filepath, "r"))


# def setup_architecure():
#     # Setting the architecture
#     reset_accelerator()
#     initialize_accelerator()
#     set_configs_to_explore()

# def setup_network():
#     # Setting the Network
#     reset_network()
#     initialize_network()


def populate_estimates(operator_list, logical_mesh):
    # setup_architecure()

    latency_estimates =  append_latency_estimates(operator_list, logical_mesh)

    # model_type = tmpc_models[0].get_model_type()
    # model_name = tmpc_models[0].model_name

    # model_estimates_dir = os.path.join(estimates_dir, model_type)
    # estimate_file_name = generate_out_filename(
    #     model_name, "json", micro_batch_size, max_tmp_width, sequence_length,ep_degree=ep_degree)

    # if only_explore_specific_configs:
    #     for model in tmpc_models:
    #         latency_estimates[str(model.tmp_width)
    #                           ] = append_latency_estimates(model)

    # else:
    #     estimate_filepath = os.path.join(
    #         model_estimates_dir, estimate_file_name)

    #     if not os.path.isfile(estimate_filepath):
    #         if not os.path.exists(model_estimates_dir):
    #             os.makedirs(model_estimates_dir)

    #         for model in tmpc_models:
    #             latency_estimates[str(model.tmp_width)
    #                               ] = append_latency_estimates(model)

    #         with open(estimate_filepath, "w") as f:
    #             try:
    #                 json.dump(latency_estimates, f,
    #                           ensure_ascii=False, indent=4)
    #             except:
    #                 os.remove(estimate_filepath)
    #                 print('\33[93m' + "Error writing to file: ",
    #                       estimate_filepath + '\033[0m')
    #         f.close

    #     else:
    #         latency_estimates = json.load(open(estimate_filepath))

    return latency_estimates

def get_wham_op_def(op):
    try:
        return wham_op_rep[op['op_name']]
    except KeyError:
        print(f'missing for this operator: {op}')
        return None


def write_global_estimates():
    with open(global_estimates_filepath, "w+") as f:
        json.dump(existing_estimates, f, ensure_ascii=False, indent=2)
    f.close()


def append_latency_estimates(operator_list, logical_mesh):
    # generate operator graphs
    # graphs with fused operators, to ensure operators across layers are not fused

    # graphs = model.get_unique_op_graphs()
    # phaze_graph = model.get_phaze_graph().get_graph()
    # fusedgraphs = [convert_phaze_to_fused_graph(graph) for graph in graphs]

    # global existing_estimates

    # estimates dictionary
    tc_estimates, vc_estimates, ar_estimates = 0,0,0

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
        total_latency = 0
        for op in operator_list:
            # print("Estimating for layer_id", list(
            #     fusedgraph.nodes.values())[0].layer_id)
            # for node in list(fusedgraph.nodes.values()):
            op_type = get_wham_op_def(op)
            if op_type == None:
                continue
            core_type = phaze_coretype_mapping[get_engine_type(
                op_type)]
            # print(f"op: {op}, core type: {core_type}")

            # fix collective ops
            if (op_type =="AllReduce"):

                # logical_mesh.all_reduce_cost(
                e = allreduce_estimator(op, logical_mesh)
                ar_estimates += e["latency"]
                total_latency += e["latency"]
                # e = allreduce_estimator(
                #     phaze_graph.nodes[node.node_id], model.tmp_width, node.node_desc, bandwidth, bytes_per_element,)
                # ar_estimates[str(node.node_id)] = e
            elif (op_type == "AllToAll"):
                e = alltoall_estimator(op, logical_mesh)
                ar_estimates += e["latency"]
                total_latency += e["latency"]
            elif (op_type == "AllGather"):
                e = allgather_estimator(op, logical_mesh)
                ar_estimates += e["latency"]
                total_latency += e["latency"]

            elif (core_type == "TC"):
                e = get_cached_result(op)
                if (e == None):
                    e = tensor_core_estimator(
                        op, TC_config, FREQUENCY)
                tc_estimates += e["latency"]
                cache_result(op, e)
                total_latency += e["latency"]

            elif (core_type == "VC"):
                # e = get_cached_result(op)
                e = vector_core_estimator(
                    op, VC_config, FREQUENCY)
                vc_estimates += e["latency"]
                # cache_result(op, e)
                total_latency += e["latency"]

        # write_global_estimates()
    except:
        # write_global_estimates()
        traceback.print_exc()
        exit("Estimates error out!")
    # print({"TC": tc_estimates, "VC": vc_estimates, "AR": ar_estimates, "Total_Latency": total_latency})

    compute_cost = tc_estimates + vc_estimates
    assert abs((compute_cost + ar_estimates) - total_latency) < 1e-5, f"compute_cost + ar_estimates {compute_cost + ar_estimates} not equal to total_latency {total_latency}"
    return {"CC": compute_cost, "AR": ar_estimates, "Total_Latency": total_latency}
