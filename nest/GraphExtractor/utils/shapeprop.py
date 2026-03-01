import torch
import torch.fx
from torch.fx.node import Node
from typing import Dict
import numpy


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod, tmp_width=1):
        self.mod = mod
        self.modules = dict(self.mod.named_modules())
        self.tmp_width = tmp_width

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}
        wrap_fn_list = [
            "get_tensor_for_fx",
            "all_reduce_for_fx_main",
            "all_reduce_for_fx_cross_entropy",
            "copy_to_tmpc_region",
            "reduce_from_tmpc_region",
            "all_to_all",
            "gather_from_sequence_parallel_region",
            "reduce_scatter_to_sequence_parallel_region",
            "MixedFusedLayerNorm",
            "FusedScaleMaskSoftmax",
        ]

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def check_for_wrap_functions(node):
            print(node.target)
            if not callable(node.target):
                return False
            print(node.target.__name__ )
            if node.target.__name__ in wrap_fn_list:
                return True
            return False

        def execute_fx_specific_all_reduce(node):
            if node.target.__name__ in [
                "all_reduce_for_fx_main",
                "all_reduce_for_fx_cross_entropy",
                "reduce_from_tensor_model_parallel_region",
                "copy_to_tensor_model_parallel_region",
            ]:
                result = load_arg(node.args)[0]
                return result
        def execute_fx_specific_all_gather(node):
            if node.target.__name__ in [
                "all_gather_context_parallel",
            ]:
                # Resolve the arguments from the graph
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                
                # 1. Get the Input Tensor (Arg 0)
                input_tensor = args[0]
                
                # 2. Get the CP Size (Arg 2)
                # The signature is: all_gather_context_parallel(tensor, group, cp_size)
                # We prioritize checking positional args[2], then fallback to kwargs['cp_size']
                if len(args) >= 3:
                    cp_size = args[2]
                elif 'cp_size' in kwargs:
                    cp_size = kwargs['cp_size']
                else:
                    # Fallback if arguments are missing (should not happen if traced correctly)
                    cp_size = 1
                print("All Gather with CP size:", cp_size)

                # 3. Simulate All-Gather by repeating the tensor
                # We repeat along dimension 0 (sequence dim) 'cp_size' times.
                # Using [1] * ndim ensures we handle 3D or 4D tensors correctly.
                repeat_dims = [1] * input_tensor.ndim
                repeat_dims[0] = cp_size
                
                return input_tensor.repeat(*repeat_dims)
            elif node.target.__name__ == "gather_from_sequence_parallel_region":
                # Infer tp_size by comparing current seq dim to original seq_len
                # input_ids placeholder is [batch, seq_len], stored in env
                # 1. Get the Input Tensor (Arg 0)
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                input_tensor = args[0]
                world_size = args[-1] if len(args) >= 2 else world_size == self.tmp_width

                loaded_kwargs = load_arg(node.kwargs)
    
                    
                print("ShapeProp: Traced All Gather with world size:", world_size)
                output_shape = list(input_tensor.shape)
                # print("shapeprop output shape before:", output_shape[0])
                output_shape[0] = output_shape[0] * world_size
                # print("shapeprop output shape after:", output_shape[0])
                return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        def execute_fx_specific_all_to_all(node):
            if node.target.__name__ in [
                "all_to_all",
            ]:
                result = load_arg(node.args)[1]
                return result
        
        def execute_fx_specific_reduce_scatter(node):
            if node.target.__name__ in ["reduce_scatter_to_sequence_parallel_region"]:
                args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)

                input_tensor = load_arg(node.args)[0]
                print("ShapeProp: Traced reduce scatter with world size:", self.tmp_width)
                
                output_shape = list(input_tensor.shape)
                output_shape[0] = output_shape[0] // self.tmp_width
                return torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        def extract_module_properties(module, node):
            if hasattr(module, "kernel_size"):
                node.kernel_size = module.kernel_size
            if hasattr(module, "padding"):
                node.padding = module.padding
            if hasattr(module, "dilation"):
                node.dilation = module.dilation
            if hasattr(module, "stride"):
                node.stride = module.stride
            if hasattr(module, "contiguous"):
                node.contiguous = module.contiguous
            if hasattr(module, "weight"):
                node.weights_shape = [list(module.weight.size())]
            if hasattr(module, "bias"):
                if isinstance(module.bias, torch.Tensor):
                    node.bias_shape = [list(module.bias.size())]

        def extract_attr_properties(target, result):
            type_tensor = target.split(".")
            if (["embedding"] in type_tensor):
                operator = "embedding"
            else:
                operator = "getattr"

            if isinstance(result, torch.Tensor):
                if type_tensor[-1] == "weight":
                    node.weights_shape = [list(result.shape)]
                if type_tensor[-1] == "bias":
                    node.bias_shape = [list(result.shape)]

            return operator


        for node in self.mod.graph.nodes:
            print(node)
            if node.op == "placeholder":
                result = next(args_iter)
                node.operator = "input"
            elif node.op == "get_attr":
                result = fetch_attr(node.target)
                node.operator = extract_attr_properties(node.target, result)
            elif node.op == "call_function":
                result = execute_fx_specific_all_reduce(node)
                if result is None:
                    result = execute_fx_specific_all_to_all(node)
                if result is None:
                    result = execute_fx_specific_all_gather(node)
                if result is None: 
                    result = execute_fx_specific_reduce_scatter(node)
                if result is None:
                    loaded_args = load_arg(node.args)
                    loaded_kwargs = load_arg(node.kwargs)
                    if node.target.__name__ == "floordiv":
                        def unwrap(val):
                            if isinstance(val, (torch.Size, tuple)) and len(val) == 1:
                                return val[0]
                            return val
                        loaded_args = tuple(unwrap(a) for a in loaded_args)
                    result = node.target(
                        *loaded_args, **loaded_kwargs)
                node.operator = node.target.__name__
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
                node.operator = node.target
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs))
                node.operator = type(self.modules[node.target]).__name__
                extract_module_properties(self.modules[node.target], node)
            elif node.op == "output":
                result = 0
                node.operator = "output"

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = [list(result.shape)]
                if hasattr(result, "tensor_model_parallel"):
                    node.tensor_model_parallel = result.tensor_model_parallel
                    node.partition_dim = result.partition_dim
                    node.partition_stride = result.partition_stride
                node.dtype = result.dtype
            elif isinstance(result, numpy.ndarray):
                node.shape = [list(result.shape)]
                node.dtype = result.dtype
            elif isinstance(result, torch.Size):
                node.shape = [list(result)]
                node.stride = tuple([1])
            elif (
                isinstance(result, int)
                or isinstance(result, bool)
                or isinstance(result, float)
                or isinstance(result, torch.finfo)
            ):
                node.shape = [[1]]
                node.stride = tuple([1])
            elif isinstance(result, tuple):
                node.shape = [[]]
                for r in result:
                    node.shape.append(list(r.shape))
                node.dtype = r[0].dtype
            elif isinstance(result, torch.dtype) or isinstance(result, torch.device) or isinstance(result, str):
                node.shape = [[]]
                node.dtype = result
            elif isinstance(result, list):
                node.shape = [[len(result)]]
                # node.dtype = result[0].dtype
                print("List", node, node.op, type(result), len(result),)
            elif result is None:
                node.shape = [[]]
                node.dtype = None
            else:
                raise TypeError("Result type not found.", node,
                                node.op, result, type(result),)

            env[node.name] = result

        return self.mod

        # return load_arg(self.graph.result)
