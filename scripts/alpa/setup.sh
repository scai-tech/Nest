#!/bin/bash
mkdir -p $CONDA_PREFIX/lib/python3.8/site-packages/jaxlib/
mkdir -p $CONDA_PREFIX/lib/python3.8/site-packages/jaxlib-0.3.22+cuda113.cudnn820.dist-info

cp -r jaxlib/*  $CONDA_PREFIX/lib/python3.8/site-packages/jaxlib/
cp -r jaxlib-0.3.22+cuda113.cudnn820.dist-info/* $CONDA_PREFIX/lib/python3.8/site-packages/jaxlib-0.3.22+cuda113.cudnn820.dist-info

cp -r transformers/*  $CONDA_PREFIX/lib/python3.8/site-packages/transformers/

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd alpa
pip3 install .
cd ..

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# Get the directory of this script dynamically
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NEST_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # adjust ../ depth to reach Nest root

echo "export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=$NEST_ROOT/scripts/alpa/alpa/alpa/estimator:\$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=$NEST_ROOT/scripts/alpa/alpa/alpa/estimator/sunstone:\$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PYTHONPATH=$NEST_ROOT/scripts/alpa/alpa/alpa/estimator/wham:\$PYTHONPATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo "Applying paths to current session..."
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh