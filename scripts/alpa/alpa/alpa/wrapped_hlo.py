"""A class that wraps HloModule and records whether the module runs AutoSharding
and SPMD Partitioner or not.
"""
from enum import Enum, auto
from typing import Union
import numpy as np

from jax._src.lib import xla_extension as xe
from jax.interpreters import mlir


class HloStatus(Enum):
    """
    The status of an HloModule.
    See also the docstring at the beginning of shard_parallel/auto_sharding.py.
    """
    UNOPTIMIZED = auto()
    SHARDING_ANNOTATED = auto()
    SPMD_PARTITIONED = auto()
    FULLY_OPTIMIZED = auto()


class WrappedHlo:
    """Wrapped HloModule with HloStatus."""

    def __init__(self,
                 module: Union[xe.HloModule, xe.XlaComputation, bytes],
                 status: HloStatus = HloStatus.UNOPTIMIZED):
        if isinstance(module, xe.HloModule):
            self.module = module
        elif isinstance(module, xe.XlaComputation):
            self.module = module.get_hlo_module()
        else:
            assert isinstance(module, bytes)
            self.module = xe.XlaComputation(module).get_hlo_module()
        self.name = self.module.name
        self.status = status
        self.is_manually_annotated = False

    def get_computation(self) -> xe.XlaComputation:
        return xe.XlaComputation(self.module.as_serialized_hlo_module_proto())

    def get_mhlo(self):
        xla_computation = self.get_computation()
        module_str = xe.mlir.xla_computation_to_mlir_module(xla_computation)
        with mlir.make_ir_context():
            mhlo = mlir.ir.Module.parse(module_str)
        return mhlo

    def get_module(self) -> xe.HloModule:
        return self.module

    def get_hlo_proto(self):
        return self.module.as_serialized_hlo_module_proto()

    def program_shape(self):
        return self.module.program_shape()

    def set_input_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_input_shardings(self.module, sharding_protos)

    def set_output_shardings(self, sharding_protos):
        assert self.is_sharding_annotated() or self.is_unoptimized()
        xe.set_hlo_module_output_shardings(self.module, sharding_protos)

    def is_unoptimized(self):
        return self.status == HloStatus.UNOPTIMIZED

    def is_sharding_annotated(self):
        return self.status == HloStatus.SHARDING_ANNOTATED

    def is_spmd_partitioned(self):
        return self.status == HloStatus.SPMD_PARTITIONED

    def to_string(self):
        return self.module.to_string()

    def __getstate__(self):
        return (self.get_hlo_proto(), self.status)

    def __setstate__(self, bytes_and_status):
        b, s = bytes_and_status
        self.__init__(b, s)

    def get_operators_and_dimensions(self):
        """
        Parses the MHLO module to extract a list of operators and their
        output tensor dimensions.
        """
        # 1. Get the parsed MLIR module
        mhlo_module = self.get_mhlo()
        op_info = []

        main_func = None
        for op in mhlo_module.body.operations:
            # Check if the operation is a function declaration and if its name is 'main'
            if op.operation.name == 'func.func':
                # print(str(op.attributes['sym_name']))
                if 'sym_name' in op.attributes and str(op.attributes['sym_name']).strip('"\'') == 'main':
                    main_func = op
                    break


        # If a main function is found, proceed with the rest of the code
        if main_func:
            op_info = []
            for op in main_func.regions[0].blocks[0].operations:
                replica_groups = None
                if op.operation.name == "mhlo.all_reduce" or op.operation.name == "mhlo.all_gather" or op.operation.name == "mhlo.all_to_all":
                    # print("Found comm op:")
                    # Access the replica_groups attribute
                    if "replica_groups" in op.operation.attributes:
                        replica_groups = []
                        try:
                            replica_groups_attr = op.operation.attributes["replica_groups"]
                            
                            # 1. Get the shape from the attribute's type.
                            # The type tells you the dimensions (e.g., [1, 512]).
                            tensor_type = mlir.ir.RankedTensorType(replica_groups_attr.type)
                            shape = tensor_type.shape
                            replica_groups_attr = mlir.ir.DenseIntElementsAttr(replica_groups_attr)
                            # np_array = np.array(replica_groups_attr)

                            # 2. Use the .elements property to get an iterator over the actual integer values.
                            # This reads the data directly, bypassing the hex representation.
                            flat_values = [val for val in replica_groups_attr]
                            # print(flat_values)

                            # 3. Reshape the flat list of values into the correct dimensions using NumPy.
                            reshaped_array = np.array(flat_values).reshape(shape)

                            # 4. Convert the NumPy array to a native Python list.

                            # 5. Now, converting this list to a string will always give the desired format.
                            replica_groups = reshaped_array.tolist()
                        except:
                            replica_groups = []

                        # 'final_string' will now be "[[0, 1, 2, ...]]"
                        # print(final_string)
                        # print(f"Found replica_groups: {replica_groups}")
                op_info.append({
                    "op_name": op.operation.name,
                    "inputs": [str(operand.type) for operand in op.operands],
                    "outputs": [str(result.type) for result in op.results],
                    "replica_groups": replica_groups
                })
            # Process op_info here
        else:
            print("Warning: No main function found with the name 'main'.")

            # # 4. For each operation, find its output tensors and their shapes
            # op_outputs = []
            # for result in op.results:
            #     # We only care about results that are tensors with a defined shape
            #     if isinstance(result.type, mlir.ir.RankedTensorType):
            #         tensor_type = mlir.ir.RankedTensorType(result.type)
            #         shape = tensor_type.shape
            #         element_type = tensor_type.element_type
            #         op_outputs.append(f"tensor<{', '.join(map(str, shape))}x{element_type}>")

            # op_info.append({
            #     "operator": op.operation.name,
            #     "outputs": op_outputs
            # })
            
        return op_info
