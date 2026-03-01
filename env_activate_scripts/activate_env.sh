export NEST_REPO_ROOT=NEST_REPO_ROOT_PLACEHOLDER  

export THIRD_PARTY_PATH=$NEST_REPO_ROOT/third_party_for_nest
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export MEGATRON_PATH=$THIRD_PARTY_PATH/nest-megatron-lm/
export ASTRA_SIM_PATH=$THIRD_PARTY_PATH/astra-sim/
export PYTHONPATH=$MEGATRON_PATH:$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$ASTRA_SIM_PATH:$PYTHONPATH
export PYTHONPATH=$NEST_REPO_ROOT/nest/Estimator:$NEST_REPO_ROOT/nest/Estimator/utils:$PYTHONPATH

export LD_LIBRARY_PATH=$THIRD_PARTY_PATH/openmpi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$THIRD_PARTY_PATH/astra-sim/extern/network_backend/ns-3/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

#!/bin/bash
# Conda activation script for nest_env

# ── 1. Identify GCC version and internal paths ──────────────────────────────
export GCC_VERSION=$($CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc -dumpversion 2>/dev/null || echo "12.4.0")
export GCC_ROOT="$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/$GCC_VERSION"
export GCC_LIBEXEC="$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/$GCC_VERSION"
export CONDA_SYSROOT="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"

# ── 2. Set Compiler Pointers ────────────────────────────────────────────────
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

# ── 3. Clear Cluster Interference ───────────────────────────────────────────
unset CPATH
unset C_INCLUDE_PATH
unset CPLUS_INCLUDE_PATH

# ── 4. Header Discovery (Python + GCC + C++) ────────────────────────────────
export PYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:$GCC_ROOT/include:$GCC_ROOT/include-fixed:$GCC_ROOT/include/c++:$GCC_ROOT/include/c++/x86_64-conda-linux-gnu"
export C_INCLUDE_PATH="$PYTHON_INCLUDE_DIR:$GCC_ROOT/include:$GCC_ROOT/include-fixed"

# ── 5. Tool & Library Discovery (Solves 'cc1plus', 'Scrt1.o', etc) ──────────
export COMPILER_PATH="$GCC_LIBEXEC:$GCC_ROOT:$COMPILER_PATH"
export PATH="$GCC_LIBEXEC:$PATH"

# Crucial: LIBRARY_PATH must include the sysroot lib64 for CRT objects
export LIBRARY_PATH="$CONDA_SYSROOT/usr/lib64:$CONDA_SYSROOT/usr/lib:$CONDA_PREFIX/lib:$GCC_ROOT:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# ── 6. Compiler Flags ───────────────────────────────────────────────────────
export CFLAGS="--sysroot=$CONDA_SYSROOT $CFLAGS"
export CXXFLAGS="--sysroot=$CONDA_SYSROOT $CXXFLAGS"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH


echo "Nest environment activated with GCC $GCC_VERSION and Python headers."