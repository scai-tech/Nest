import torch

from torch.fx import Tracer
from torch.fx.graph_module import GraphModule

try:
    from megatron.training import get_args
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
    from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax, ScaledSoftmax, ScaledMaskedSoftmax
    from megatron.legacy.model.fused_layer_norm import MixedFusedLayerNorm
    from megatron.legacy.model.fused_softmax import FusedScaleMaskSoftmax as legacyFusedScaleMaskSoftmax
    from megatron.legacy.model.fused_softmax import ScaledSoftmax as legacyScaledSoftmax
    from megatron.legacy.model.fused_bias_gelu import GeLUFunction as legacyGeLUFunction
    from megatron.core.fusions.fused_bias_gelu import GeLUFunction
    from megatron.core.tensor_parallel.mappings import _CopyToModelParallelRegion, _ReduceFromModelParallelRegion, _AllToAll, _GatherFromSequenceParallelRegion, _ScatterToSequenceParallelRegion
except:
    print("\33[93m" + "Megatron is not installed. Megatron models will not be supported for full import." + "\033[0m")
    pass


class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """

    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(m, torch.nn.Sequential)


# class ArangeForFx(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self._is_leaf_module = True

#     def forward(self, x):
#         return torch.arange(x)

class ArangeForFx(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_leaf_module = True

    def forward(self, start=0, end=None, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False):
        return torch.arange(start=start, end=end, step=step, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)


class EmptyForFx(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_leaf_module = True

    def forward(self, x):
        return torch.empty(x)


def custom_tracer_for_megatron(model, concrete_args=None):
    tracer = CustomedTracer(
        customed_leaf_module=(
            _ReduceFromModelParallelRegion,
            _CopyToModelParallelRegion,
            _AllToAll,
            ArangeForFx,
            FusedLayerNorm,
            MixedFusedLayerNorm,
            EmptyForFx,
            ScaledSoftmax,
            FusedScaleMaskSoftmax,
            GeLUFunction,
            legacyFusedScaleMaskSoftmax, 
            legacyScaledSoftmax, 
            legacyGeLUFunction, 
            ScaledMaskedSoftmax,
            _ScatterToSequenceParallelRegion,
            _GatherFromSequenceParallelRegion
        )
    )
    graph = tracer.trace(model, concrete_args=concrete_args)
    name = model.__class__.__name__ if isinstance(
        model, torch.nn.Module) else model.__name__
    traced = GraphModule(tracer.root, graph, name)

    return traced
