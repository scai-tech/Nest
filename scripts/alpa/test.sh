
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
cd alpa
export GRPC_VERBOSITY=ERROR
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
ray start --head
python3 -m alpa.test_install
ray stop
cd ..