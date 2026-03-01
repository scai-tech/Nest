# internal code import
from ..utils import PhazeGraph
from ..utils import language_models, megatron_language_models, mixtral_models
from ..utils import store_obj_to_file, load_obj_from_file

supported_models = language_models


class BaseModelIR:
    def __init__(self, model_name, tmp_width=1, ep_degree=1, sp_enabled=False, cp_degree=1) -> None:
        self.model = None
        self.summary = None
        self.tmp_width = tmp_width
        self.model_name = model_name
        self.ep_degree = ep_degree
        self.sp_enabled = sp_enabled
        self.cp_degree = cp_degree

        if (ep_degree > 1 or cp_degree > 1 or tmp_width > 1 or sp_enabled):
            assert model_name in megatron_language_models, "TMP, EP, CP, and SP only supported with Megatron Models, currently not support with HuggingFace models"

        if (sp_enabled):
            assert tmp_width > 1, "Sequence Parallel can only be enabled if TMP degree > 1 for Megatron Models"
        
        if (ep_degree > 1):
            assert model_name in mixtral_models, "Expert Parallel can only work for MoE models"
        

        self.graphmodule = None
        self.phazegraph = PhazeGraph(self.tmp_width, self.model_name, self.ep_degree, self.sp_enabled, self.cp_degree )

        # source of model graph and trace - either extracted model or obtained only the trace from file
        self.trace_only_model = None

        # These values are set when model estimator is executed
        # model tensor properties
        self.model_inputs = None
        self.number_of_parameters = 0
        self.model_outputs = None

    def get_phaze_graph(self):
        return self.phazegraph

    def get_op_graph(self, layer_id):
        return self.phazegraph.get_op_layer_graph(layer_id)

    def get_op_graphs(self):
        return self.phazegraph.get_op_layer_graphs()

    def get_unique_op_graphs(self):
        return self.phazegraph.get_unique_op_graphs()

    def get_layer_ids(self):
        return self.phazegraph.get_layer_ids()

    def get_layer_graph(self):
        return self.phazegraph.get_layer_graph()

    def is_layer_info_extended(self):
        try:
            return self.phazegraph.layer_info_extended
        except:
            return False

    def create_graph_from_symbolic_trace(self):
        curr_layer_id = -1
        graphmodule = self.graphmodule.graph

        layer_ids = set()

        # create nodes in the graph
        g = self.get_phaze_graph()

        for n in graphmodule.nodes:
            (if_layer_identifier, layer_id) = self.get_layer_id(n, curr_layer_id)
            if if_layer_identifier:
                curr_layer_id = layer_id
            g.setup_node_and_edges(graphmodule, n, curr_layer_id)

            # add layer information
            layer_ids.add(curr_layer_id)

        g.set_layer_ids(layer_ids)

    def get_repeat_layer_ids(self):
        g = self.get_phaze_graph()
        return g.get_repeat_layer_ids()

    def generate_layer_info(self):
        g = self.phazegraph

        if g is None:
            raise ValueError(
                "Model for model name" + self.model_name,
                "and tensor model parallel width" + self.tmp_width + "does not exist",
            )

        g.set_repeat_layer_ids(language_models)
        g.generate_layer_info()
        g.set_op_layer_graphs()
        g.contract_layer_graph()

    # This function generates the memory footprint for layer/layer slice, latter if tmp_width > 1
    # 1. weights (times 2, plus optimizer state)
    # 2. peak memory usage, DEFINED AS sum of output activations across all nodes in the layer/layer slice (ergo, equal for fw and bw pass)
    # 3. input size to each layer/layer slice

    def get_layerwise_memory_footprint(self):
        g = self.phazegraph
        return g.get_layerwise_memory_footprint()

    # This function accumulates memory footprint for the entire model
    def get_memory_footprint(self):
        g = self.phazegraph

        print(g.get_number_of_parameters())

        return g.get_memory_footprint()

    def set_model(self):
        raise NotImplementedError("Must override set_model")

    def obtain_symbolic_trace_model(self, micro_batch_size, sequence_length):
        raise NotImplementedError("Must override obtain_symbolic_trace_model")

    def get_model_type(self):
        raise NotImplementedError("Must override get_out_dir")

    def get_out_dir(self):
        raise NotImplementedError("Must override get_out_dir")

    def get_layer_id(self, node, curr_layer_id=-1):
        raise NotImplementedError("Must override get_layer_id")

    def load_language_model(self, out_dir, micro_batch_size=1, sequence_length=64, force_reextract_model=False, ep_degree=1, sp_enabled=False, cp_degree=1):

        if self.tmp_width is None:
            raise ValueError("Tensor Model Width is set as None")

        phazegraph = None
        if not force_reextract_model:
            phazegraph = load_obj_from_file(
                self.model_name, micro_batch_size, self.tmp_width, sequence_length, module_type="graph", ep_degree=ep_degree, sp_enabled=sp_enabled, cp_degree=cp_degree)

        if isinstance(phazegraph, PhazeGraph):
            print("Loaded phaze graph successfully from file for model",
                  self.model_name,)
            self.phazegraph = phazegraph
            # this implies self.model and self.graphmodule is None but phazegraph is populated
            self.trace_only_model = True
        else:
            print("not the right instance",
                  self.model_name,)
            self.set_model()
            self.obtain_symbolic_trace_model(micro_batch_size, sequence_length)
            self.create_graph_from_symbolic_trace()
            store_obj_to_file(self.model_name, self.phazegraph, micro_batch_size,
                              self.tmp_width, sequence_length, module_type="graph",ep_degree=self.ep_degree, sp_enabled=self.sp_enabled, cp_degree=self.cp_degree)

        self.generate_layer_info()

        g = self.get_phaze_graph()

        # if g.get_num_nodes() < 1000:
        # g.print_graph(out_dir, micro_batch_size, sequence_length)
        #g.print_graph(out_dir, micro_batch_size, sequence_length)

        # print the memory of the model
        model_memory = g.get_memory_footprint()
        print(g.get_layerwise_memory_footprint())
        print(g.get_memory_footprint())
        # print("Model parameter size", model_memory.parameter_size)

        # g.print_layer_graph(out_dir, micro_batch_size, sequence_length)

        # storing entire models can take a lot of space
        # store_obj_to_file(self.model_name, self, micro_batch_size,
        #                  self.tmp_width, sequence_length,)
