# from .graph_wrapper import construct_external_scheduler, get_engine_type
from alpa.estimator.tc_estimator import tensor_core_estimate

# Third party imports
from collections import namedtuple

# Python imports
from math import inf, log2, prod
import sys
import os
import time
import numpy as np #NEST
import re #NEST
import ast # Using ast.literal_eval is safer than eval() # NEST

from initialize import initialize as hardware_initializer
from initialize import get_core_energy as get_external_core_energy
from initialize import get_core_area as get_external_core_area
from initialize import run_area_energy_generation

from alpa.estimator.wham.perf_wrappers.tensor_core_wrapper import get_dims_and_fused_perf_est as get_tc_dims
from alpa.estimator.wham.perf_wrappers.vector_core_estimator import get_performance_est as external_vc_estimator

# from .network import get_allreduce_latency
# from .network import get_alltoall_latency

##############################################################################################
############################## Accelerator Setup and Initialization ##########################
##############################################################################################

acc_setup_status = {}
acc_setup_status["dir_setup"] = False
acc_setup_status["setup_dir_path"] = None
acc_setup_status["config_setup"] = False
acc_setup_status["curr_config"] = None

tmpMemorization_allreduce = {}
tmpMemorization_alltoall = {}

use_peak = False
bytes_per_element = 2

phaze_coretype_mapping = {"Tensor Core": "TC", "Vector Core": "VC",
                          "Tensor Core + Vector Core": "TCandVC", "Nop": "Nop"}

e_tuple = namedtuple(
    "estimate", ["latency", "energy", "utilization", "estimation_time"])

wham_op_rep = {
    "mhlo.dot": "Mm",
    "mhlo.dot_general": "Mm",
    "mhlo.add": "Add",
    "mhlo.broadcast_in_dim": "Expand",
    "mhlo.conv": "Conv",
    "mhlo.reshape": "View",
    "mhlo.maximum": "Index",
    "mhlo.select": "Index",
    "mhlo.subtract": "Sub",
    "mhlo.multiply": "Mul",
    "mhlo.compare": "View",
    "mhlo.tuple": "View",
    "mhlo.constant": None,
    "mhlo.reduce": "Sum",
    "mhlo.transpose": "Transpose",
    "func.return": None,
    "mhlo.negate": "Neg",
    "mhlo.divide": "Div",
    "mhlo.scatter": "MaskedFill",
    'mhlo.tanh': "Tanh",
    "mhlo.exponential": "Pow",
    "mhlo.iota" : "MaskedFill",
    "mhlo.convert": "Transpose",
    "mhlo.pad": "Expand",
    "mhlo.slice" : "Slice",
    "mhlo.rsqrt": "Sqrt",
    "mhlo.sqrt": "Sqrt",
    "mhlo.clamp": "Index",
    "mhlo.concatenate" : "Expand",
    "mhlo.and": "Index",
    "mhlo.dynamic_update_slice": "Slice",
    "mhlo.partition_id": None,
    'mhlo.dynamic_slice': "Slice",
    "mhlo.gather": "Slice",
    # Communication ops
    "mhlo.all_reduce": "AllReduce",
    "mhlo.all_gather": "AllGather",
    "mhlo.all_to_all": "AllToAll",
    "mhlo.collective_permute" : "AllToAll", # Similar to sparse all to all, similar communication cost

}
def initialize_accelerator():
    def dir_exists_or_create(dirpath):
        isExist = os.path.exists(dirpath)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dirpath)
            Warning("The", dirpath, "directory for architectural explorations!")

    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    import time

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "tmp")

    gemm_dir = os.path.join(tmp_dir, "GEMM")
    vc_dir = os.path.join(tmp_dir, "vector_core")
    estimates_tc_dir = os.path.join(gemm_dir, "arch_estimates")
    estimates_vc_dir = os.path.join(vc_dir, "arch_estimates")

    dir_exists_or_create(tmp_dir)
    dir_exists_or_create(vc_dir)
    dir_exists_or_create(gemm_dir)

    config_dir = os.path.join(os.path.dirname(curr_dir), "arch_configs/* ")

    os.system("cp -r " + str(config_dir) + str(tmp_dir))

    dir_exists_or_create(estimates_tc_dir)
    dir_exists_or_create(estimates_vc_dir)

    acc_setup_status["dir_setup"] = True
    acc_setup_status["setup_dir_path"] = tmp_dir
    acc_setup_status["setup_tc_dir_path"] = gemm_dir
    acc_setup_status["setup_vc_dir_path"] = vc_dir

    print("Initialized accelerator directories and files for area and energy estimation!")


def reset_accelerator():
    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "tmp")

    if (os.path.exists(tmp_dir)):
        os.system("rm -rf " + str(tmp_dir))

    acc_setup_status["dir_setup"] = False
    acc_setup_status["setup_dir_path"] = None
    acc_setup_status["config_setup"] = False
    acc_setup_status["curr_config"] = None


# Sets up the architecture yaml files in tmp/GEMM/arch/arch.yaml and tmp/vector_core/arch/arch.yaml
def setup_accelerator_with_config(config):
    acc_setup_status["curr_config"] = config
    acc_setup_status["config_setup"] = True

    assert acc_setup_status["dir_setup"] == True
    assert acc_setup_status["setup_dir_path"] is not None
    assert acc_setup_status["setup_tc_dir_path"] is not None
    assert acc_setup_status["setup_vc_dir_path"] is not None

    hardware_initializer(config, acc_setup_status["setup_dir_path"])

def initialize_network():
    def dir_exists_or_create(dirpath):
        isExist = os.path.exists(dirpath)

        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(dirpath)
            Warning("The", dirpath, "directory for network explorations!")

    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "utils/tmp/network")

    net_ip_dir = os.path.join(tmp_dir, "inputs")

    dir_exists_or_create(tmp_dir)
    dir_exists_or_create(net_ip_dir)

    net_setup_status["dir_setup"] = True
    net_setup_status["setup_dir_path"] = tmp_dir
    net_setup_status["setup_ip_dir"] = net_ip_dir

    print("Initialized network directories and files for astra sim!")

    # curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")

    # tmp_net_dir = os.path.join(os.path.dirname(curr_dir), "utils/tmp/network")

def reset_network():
    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    tmp_dir = os.path.join(os.path.dirname(curr_dir), "utils/tmp/network")

    if (os.path.exists(tmp_dir)):
        os.system("rm -rf " + str(tmp_dir))

    net_setup_status["dir_setup"] = False
    net_setup_status["setup_dir_path"] = None
    net_setup_status["config_setup"] = False
    net_setup_status["curr_config"] = None

##################################### Estimator Functions #####################################

# Convert the core configuration for the estimator functions


def create_core_config_for_estimation(core_config, ):
    max_glb_size = 29360128
    max_glb_bw = 4096
    buffer_scale_metric = 256

    dim1 = 2**int(log2(core_config.num)/2)
    dim2 = core_config.num // dim1

    curr_config_factor = max(
        core_config.width, core_config.depth) * max(dim1, dim2)

    if (core_config.num == 1):
        GLB_BW = min(max(max_glb_bw * curr_config_factor /
                         buffer_scale_metric, 4), max_glb_bw)
    else:
        GLB_BW = max_glb_bw

    L2_Buffer = (2**(log2(core_config.width) +
                 log2(core_config.depth) - 6)) * 1024
    if (L2_Buffer < 1024):
        L2_Buffer = 1024

    acc_config = {
        # Compute Unit configuration
        "Core_x": max(dim1, dim2),
        "Core_y": min(dim1, dim2),
        "PE_x": core_config.width,
        "PE_y": core_config.depth,
        "VC_PE": core_config.width,
        # Global Buffer bandwidth
        "GLB_BUFFER_BW": GLB_BW,
        # GLB_Buffer size
        "GLB_Buffer": core_config.GLB_Buffer,
        "L2_Buffer": L2_Buffer,
        # L1_Buffer size
        "L1_Buffer_TC": 32,
        "L1_Buffer_VC": 12,
        # Dataflow and Skipping
        "GLB_Buffer_skip": "Weights",
        "L2_Buffer_skip": "Inputs, Weights, Outputs",
        "L1_Buffer_skip": "None",
    }

    return acc_config


# Sets up the architecture yaml files in tmp/tensorcore/GEMM/arch
# Generates the area estimate for the setup architecture in tmp/tensorcore/GEMM/arch_estimates/ART.yaml

def get_core_area(config, core_type):
    setup_config = create_core_config_for_estimation(config)
    if acc_setup_status["curr_config"] != setup_config or acc_setup_status["config_setup"]:
        setup_accelerator_with_config(setup_config)

    if (core_type == "TC"):
        num_macs = (setup_config["Core_x"] * setup_config["Core_y"]) * \
            (setup_config["PE_x"] * setup_config["PE_y"])
        dir = acc_setup_status["setup_tc_dir_path"]
    if (core_type == "VC"):
        num_macs = (setup_config["Core_x"] *
                    setup_config["Core_y"]) * setup_config["VC_PE"]
        dir = acc_setup_status["setup_vc_dir_path"]

    return get_external_core_area(setup_config, num_macs, dir)


# Checks if the config is the current config being used
# Generates the energy estimate for the setup architecture in tmp/tensorcore/GEMM/arch_estimates/ERT_summary.yaml
def get_core_energy(coreconfig, core_type):
    setup_config = create_core_config_for_estimation(coreconfig)
    if core_type == "TC":
        dir = acc_setup_status["setup_tc_dir_path"]
    elif core_type == "VC":
        dir = acc_setup_status["setup_vc_dir_path"]
    else:
        raise ValueError("Invalid core type")

    if acc_setup_status["curr_config"] != setup_config or acc_setup_status["config_setup"]:
        setup_accelerator_with_config(setup_config)

    run_area_energy_generation(dir)
    return get_external_core_energy(dir)


def generate_fwd_bwd_dims(wgraph, wnode, setupconfig, estimates_file=""):
    schd = construct_external_scheduler(wgraph, setupconfig, estimates_file)
    return get_tc_dims(schd, wnode)


def get_numpy_shapes_from_list(tensor_list: list):
    """
    Parses a list of tensor strings (either 'inputs' or 'outputs' from the
    original dictionary entry) and returns a list of numpy array shapes.

    Args:
        tensor_list (list): A list of strings, where each string is a tensor
                            representation (e.g., 'tensor<128x2048xf32>').
                            This can be the list from the 'inputs' key or the
                            list from the 'outputs' key (which contains a single
                            'tuple<...>' string).

    Returns:
        list[np.ndarray]: A list of numpy array shapes corresponding to the
                          tensors in the list.

    Args:
        tensor_list (list): A list of strings, where each string is a tensor
                            representation (e.g., 'tensor<128x2048xf32>').
                            This can be the list from the 'inputs' key or the
                            list from the 'outputs' key (which contains a single
                            'tuple<...>' string).

    Returns:
        list[np.ndarray]: A list of numpy array shapes corresponding to the
                          tensors in the list.
                          
    Example:
    >>> inputs = ['tensor<128x2048xf32>', 'tensor<128x2048xi1>']
    >>> get_numpy_shapes_from_list(inputs)
    [[128, 2048], [128, 2048]]
    >>> outputs = ['tuple<tensor<128x2048xf32>, tensor<128x2048xi1>>']
    >>> get_numpy_shapes_from_list(outputs)
    [[128, 2048], [128, 2048]]
    """
    if not tensor_list:
        return []

    shapes = []
    
    # Check if the list contains a tuple string (like the 'outputs' key)
    if tensor_list[0].startswith('tuple<'):
        # For 'outputs', we first need to extract the content inside the tuple<>
        tuple_content = tensor_list[0]
        # Find all the individual tensor<> strings inside the tuple
        tensor_strings = re.findall(r'tensor<[^>]+>', tuple_content)
    else:
        # Otherwise, assume it's a simple list of tensor strings (like 'inputs')
        tensor_strings = tensor_list
        
    for tensor_str in tensor_strings:
        # For each tensor string, find the numerical dimensions
        dims_str = re.findall(r'\d+', tensor_str)
        # The dimensions are all parts except for the very last one.
        dims_str = dims_str[:-1]
        # Convert the found strings to integers
        dims = [int(d) for d in dims_str]
        shapes.append(dims)
            
    return shapes


def parse_comm_group_strings(string):
    """
    Parses a list of strings to extract numerical data from 'dense<...>' 
    tags and returns a list of NumPy arrays.

    Args:
        string_list (list): A list of strings to parse.

    Returns:
        list: A list containing NumPy arrays or empty lists for special cases.
    """
    if not string:
        return []
    if isinstance(string, list):
        # If the input is a list, we need to process each string in the list
        return string
        
    else:
        string = str(string)

    results = []
    # This regular expression looks for the pattern 'dense<' followed by
    # any characters that are not '>', captures them, and then matches the closing '>'.
    pattern = re.compile(r'dense<([^>]*)>')

    match = pattern.search(string)
    if not match:
        results = []
        return results

    content_str = match.group(1)

    # Handle the special case "dense<0>" which should be an empty list
    if content_str == '0':
        results = []
        return results
    
    # Use ast.literal_eval to safely evaluate the string as a Python literal (e.g., a list)
    # It's a secure alternative to the built-in eval() function.
    try:
        data_list = ast.literal_eval(content_str)
        list_of_1d_arrays = [np.array(sublist) for sublist in data_list]
        results = list_of_1d_arrays
    except (ValueError, SyntaxError):
        # Handle cases where the content is not a valid Python literal
        print(f"Could not parse content: {content_str}")
        results = []
            
    return results



def get_input_output(op):
    input_shapes = get_numpy_shapes_from_list(op["inputs"])
    # print("Input shapes:")
    # print(op["inputs"])
    # print(input_shapes)

    output_shapes = get_numpy_shapes_from_list(op["outputs"])
    # print("\nOutput shapes:")
    # print(op["outputs"])
    # print(output_shapes)
    return input_shapes, output_shapes


def get_problem_dim(op, batch_size=1):

    name = wham_op_rep.get(op["op_name"])
    input_shapes, output_shapes = get_input_output(op)
    output_act = output_shapes[0]
    input_act = input_shapes[0]
    weight = input_shapes[1]

    if (
        name=="Conv"
    ):

        N = output_act[0]
        M = output_act[1]
        P = output_act[2]
        Q = output_act[3]

        W = input_act[2]
        H = input_act[3]

        C = weight[1]
        R = weight[2]
        S = weight[3]

        B = 1

        W_stride = 1 # to fix NEXT
        H_stride = 1
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        Type = "CONV"

    elif (name=="Mm"):

        B = batch_size

        # For Outer, which is basically a (M x 1) (1 x N) matmul

        if len(weight) == 1:
            weight.insert(0, 1)
        if len(input_act) == 1:
            input_act.append(1)

        N = output_act[0]
        M = output_act[1]
        P = 1
        Q = 1

        W = 1
        H = 1

        C = weight[0]
        R = 1
        S = 1

        Type = "CONV"

        W_stride = 1
        H_stride = 1
        W_dilation = 0
        H_dilation = 0
        W_padding = 0
        H_padding = 0

        """
        if C <= definitions.TC_PE_x:
            C = C
        else:
            C = math.ceil(C/definitions.TC_PE_x) * definitions.TC_PE_x
        
        if M <= definitions.TC_PE_y:
            M = M
        else:
            M = math.ceil(M/definitions.TC_PE_y) * definitions.TC_PE_y
        """

        # assert (
        #         output_act[0] == input_act[0]
        #         and output_act[1] == weight[1]
        #         and input_act[1] == weight[0]
        #     ), "Dimensions doesn't match!!" + str((output_act, input_act, weight))
    
    return (
        B,
        N,
        M,
        C,
        W,
        H,
        R,
        S,
        W_stride,
        H_stride,
        W_dilation,
        H_dilation,
        Type,
        P,
        Q,
        W_padding,
        H_padding,
    )

    
    


def tensor_core_estimator(op, core_config, f=10 ** 6):
    setup_config = create_core_config_for_estimation(core_config)
    # dims, fused_cyles = generate_fwd_bwd_dims(wgraph, wnode, setup_config)

    dims = get_problem_dim(op)

    (
        b,  # batch size
        n, # output dim[0]
        m,  # output dim [1]
        c, # weight dim[0]
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
    ) = dims

    # bypass vocab size large calculations

    if m == 30522 or c == 30522 or n == 30522 or m == 30522 or c == 15261 or n == 15261:#bert
        return {"latency": 0}
    elif m == 32000 or c == 32000 or n == 32000 or m == 16000 or c == 16000 or n == 16000: #llama2
        return {"latency": 0}
    elif m == 128256 or c == 128256 or n == 128256 or m == 64128 or c == 64128 or n == 64128 or m == 32064 or c == 32064 or n == 32064  or  m == 32064 or c == 32064 or n == 32064 or  m == 16032 or c == 16032 or n == 16032: #llama3
        return {"latency": 0}
    elif m == 50257 or c == 50257 or n == 50257: #gpt3
        return {"latency": 0}

    energy = []

    # setup_accelerator_with_config(setup_config)

    fwd_tuple = tensor_core_estimate(setup_config, dims, energy)

    if fwd_tuple is None:
        efwd = e_tuple(inf, 0, 0, 0,)
        fwd_latency = inf
    else:
        fwd_latency = (fwd_tuple[0]) / f
        efwd = e_tuple(
            fwd_latency, -1, -1, 0,)
        
    # rd_wr_fnc(core_config, dims[0], efwd)
    # rd_wr_fnc(core_config, dims[1], ebwd)

    # print(op, efwd)
    if fwd_latency > 1.0:
        print(f" tensor (Large latency) dimensions: {dims},  latency: {fwd_latency}")

    return {"latency":fwd_latency}


def vector_core_estimator(op, core_config, f=10 ** 6):
    setup_config = create_core_config_for_estimation(core_config)
    # scheduler = construct_external_scheduler(wgraph, setup_config, "")

    name = wham_op_rep.get(op["op_name"])
    input_shapes, output_shapes = get_input_output(op)

    start = time.time()
    # setup_accelerator_with_config(setup_config)
    energy = []
    fwd_latency = external_vc_estimator(name, input_shapes, output_shapes, setup_config, energy) / f
    end = time.time()
    efwd = e_tuple(fwd_latency, -1, -1, 0,)

    # efwd = e_tuple(wnode.fwd_latency / f, wnode.fwd_energy /
    #                (10**12), -1, end - start,)
    # ebwd = e_tuple(wnode.bwd_latency / f, wnode.fwd_energy /
    #                (10**12), -1, end - start,)

    # print(f" vector name: {name},  latency: {fwd_latency}")

    return {"latency":fwd_latency}

def get_cycles(dataSize, num_devices, operation):
    key = (dataSize, num_devices)
    if(operation == "AllReduce"):
        if key in tmpMemorization_allreduce:
            # print(f"Cache hit for size={dataSize} and num_devices={num_devices}")
            return tmpMemorization_allreduce[key]
        else:
            # print(f"Cache miss for size={dataSize} and num_devices={num_devices}")
            result = get_allreduce_latency(list(range(num_devices)), dataSize)
            tmpMemorization_allreduce[key] = result  # Store in the cache
            return result
    elif(operation == "AllToAll"):
        if key in tmpMemorization_alltoall:
            # print(f"Cache hit for size={dataSize} and num_devices={num_devices}")
            return tmpMemorization_alltoall[key]
        else:
            # print(f"Cache miss for size={dataSize} and num_devices={num_devices}")
            result = get_alltoall_latency(list(range(num_devices)), dataSize)
            tmpMemorization_alltoall[key] = result  # Store in the cache
            return result

def allreduce_estimator(op, logical_mesh):
    input_shapes, output_shapes = get_input_output(op)
    comm_group_list = op["replica_groups"]

    if len(comm_group_list) == 0:
        dims = [0 , 1]
    else:
        try:
            devices = len(comm_group_list[0])
            # print(f"devices: {comm_group_list[0]}, the string: {op['replica_groups']}")
        except:
            print(f"could not get devices: {comm_group_list[0]}, the string: {op['replica_groups']}")
        group_list = comm_group_list[0]
        if devices == 1 and group_list[0] == 0: # [[0]] means full group all reduce
            dims = [0 , 1]
        elif devices == 1:
            dims = [] # only single device, noop
        elif devices == logical_mesh.num_devices:
           dims = [0 , 1]
        elif devices <= logical_mesh.id_mesh.shape[1] and logical_mesh.id_mesh.shape[1] > 1 and group_list[1] - group_list[0] == 1: # NEST:  consecutive device num == same host
            dims = [1]
        elif devices <= logical_mesh.id_mesh.shape[0] and logical_mesh.id_mesh.shape[0] > 1: # across different "logical" hosts
            dims = [0]
    # print(f"AllReduce dims: {dims}")

    tensorsize = np.prod(output_shapes[0]) * bytes_per_element
    # assume reduce done on both dimensions
    fwd_latency = 0
    for dim in dims:
        fwd_latency += logical_mesh.all_reduce_cost(tensorsize, dim)

    return {"latency":fwd_latency}

def alltoall_estimator(op, logical_mesh):
    

    input_shapes, output_shapes = get_input_output(op)
    comm_group_list = op["replica_groups"]
    
    dims = []

    if comm_group_list == None or len(comm_group_list) == 0:
        dims = [0 , 1]
    else:
        try:
            devices = len(comm_group_list[0])
        except:
            print(f"could not get devices: {comm_group_list[0]}, the string: {op['replica_groups']}")
        group_list = comm_group_list[0]
        if devices == 1 and group_list[0] == 0: # [[0]] means full group all reduce
            dims = [0 , 1]
        elif devices == 1:
            dims = [] # only single device, noop
        elif devices == logical_mesh.num_devices:
           dims = [0 , 1]
        elif devices <= logical_mesh.id_mesh.shape[1] and logical_mesh.id_mesh.shape[1] > 1 and group_list[1] - group_list[0] == 1: # NEST:  consecutive device num == same host
            dims = [1]
        elif devices <= logical_mesh.id_mesh.shape[0] and logical_mesh.id_mesh.shape[0] > 1: # across different "logical" hosts
            dims = [0]
    # print(f"AlltoAll op: {op}, comm_group_list: {comm_group_list}, dims: {dims}")
    # print(f"AlltoAll dims: {dims}")


    tensorsize = np.prod(input_shapes[0]) * bytes_per_element
    # assume reduce done on both dimensions
    fwd_latency = 0
    for dim in dims:
        fwd_latency += logical_mesh.all_to_all_cost(tensorsize, dim)
    # print(f"AlltoAll comm_group_list: {comm_group_list}, latency:{fwd_latency}")

    return {"latency":fwd_latency}


def allgather_estimator(op, logical_mesh):
    input_shapes, output_shapes = get_input_output(op)
    comm_group_list = op["replica_groups"]

    if len(comm_group_list) == 0:
        dims = [0 , 1]
    else:
        try:
            devices = len(comm_group_list[0])
        except:
            print(f"could not get devices: {comm_group_list[0]}, the string: {op['replica_groups']}")
        group_list = comm_group_list[0]
        if devices == 1 and group_list[0] == 0: # [[0]] means full group all reduce
            dims = [0 , 1]
        elif devices == 1:
            dims = [] # only single device, noop
        elif devices == logical_mesh.num_devices:
           dims = [0 , 1]
        elif devices <= logical_mesh.id_mesh.shape[1] and logical_mesh.id_mesh.shape[1] > 1 and group_list[1] - group_list[0] == 1: # NEST:  consecutive device num == same host
            dims = [1]
        elif devices <= logical_mesh.id_mesh.shape[0] and logical_mesh.id_mesh.shape[0] > 1: # across different "logical" hosts
            dims = [0]
    # print(f"AllGather dims: {dims}")

    tensorsize = np.prod(input_shapes[0]) * bytes_per_element
    # assume reduce done on both dimensions
    fwd_latency = 0
    for dim in dims:
        fwd_latency += logical_mesh.all_gather_cost(tensorsize, dim)

    return {"latency":fwd_latency}


def get_flops(fusedgraph):
    VC_FLOPS = 0
    TC_FLOPS = 0

    for node in list(fusedgraph.nodes.values()):
        core_type = phaze_coretype_mapping[get_engine_type(
            node.node_desc)]

        if (core_type == "TC" or core_type == "TCandVC"):
            dims, fused_cyles = generate_fwd_bwd_dims(fusedgraph, node, {})
            TC_FLOPS += prod([i for i in dims[0] if type(i) == int and i != 0])
            TC_FLOPS += prod([i for i in dims[1] if type(i) == int and i != 0])

        if (core_type == "VC" or core_type == "TCandVC"):
            # for fwd and backward pass
            VC_FLOPS += 2 * prod(node.output_act[0])

    print("Number of nodes", len(fusedgraph.nodes.values()))
    print("VC FLOPS: ", VC_FLOPS / (10**9))
    print("TC FLOPS: ", TC_FLOPS / (10**9))


def calc_lat_peak_flops(setup_config, op):

    (b,
        n,
        m,
        c,
        w,
        h,
        r,
        s,
        w_stride,
        h_stride,
        w_dilation,
        h_dilation,
        types,
        p,
        q,
        w_padding,
        h_padding,
     ) = op

    '''dim = {
        "N": n * b,  # batch size #B
        "C": c, #N
        "M": m, #C
        "R": r, #r
        "S": s, #s
        "P": p, #p
        "Q": q, #q
    }'''

    dim = {
        "N": n * b,  # batch size
        "C": c,
        "K": m,
        "R": r,  # r
        "S": s,  # s
        "P": p,  # p
        "Q": q,  # q
    }

    numOp = 2 * n * b*c*m*r*s*p*q
    numMacs = (setup_config["Core_x"] * setup_config["Core_y"]) * \
        (setup_config["PE_x"] * setup_config["PE_y"])
    num_peak_flop_cycle = numMacs * 2

    return (numOp / (num_peak_flop_cycle / 2), 0, 0)
