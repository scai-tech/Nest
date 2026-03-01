#!/bin/bash

############################################
# # Full install
# bash setup.sh

# # Apex only
# bash setup.sh --apex-only

#################################################

# Flag parsing
APEX_ONLY=false

for arg in "$@"; do
  case $arg in
    --apex-only)
      APEX_ONLY=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# ─────────────────────────────────────────────
# Full install (skipped if --apex-only is set)
# ─────────────────────────────────────────────

export CXX=$(which g++)
export CC=$(which gcc)

if [ "$APEX_ONLY" = false ]; then

  # Dynamically find the repo root based on conda env location
  NEST_REPO_ROOT=$(pwd)
  echo "Setting conda paths: $NEST_REPO_ROOT for the conda env $CONDA_PREFIX"

  echo "Installing activation scripts..."
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d
  mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

  # Bake the repo root path into the activation script
  sed "s|NEST_REPO_ROOT_PLACEHOLDER|$NEST_REPO_ROOT|g" env_activate_scripts/activate_env.sh \
  > $CONDA_PREFIX/etc/conda/activate.d/nest_paths.sh

  cp env_activate_scripts/deactivate_env.sh $CONDA_PREFIX/etc/conda/deactivate.d/nest_paths.sh
  echo "Done! Re-activate your env to apply paths: conda activate nest_env"

  echo "Applying paths to current session..."
  source $CONDA_PREFIX/etc/conda/activate.d/nest_paths.sh

  ######################################################

   # Megatron specific installations
  ######################################################
  pip3 install git+https://github.com/NVIDIA/Megatron-LM.git --no-cache-dir
  cp -r third_party_for_nest/nest-megatron-lm/megatron/* $CONDA_PREFIX/lib/python3.10/site-packages/megatron/

  pip uninstall torch -y

  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir


  ######################################################

  git clone https://github.com/Accelergy-Project/accelergy.git
  cd accelergy
  git checkout 0278a565187dc019ca40043ed486bf94b645327e
  pip3 install .
  accelergy
  cd ..

  git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
  cd accelergy-cacti-plug-in
  git checkout ba5468303c27b4a1a317742a4eaf147065b907e5
  pip3 install .

  git clone https://github.com/HewlettPackard/cacti.git
  cd cacti
  make
  export PATH=$(pwd):${PATH}
  cd ../..

  git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
  cd accelergy-table-based-plug-ins/
  git checkout 223039ffbf0e034f3b09c2b80074ad398fbaf03e
  pip3 install . --no-build-isolation
  cd ..

  git clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git
  cd accelergy-library-plug-in/
  git checkout 0cab62c3631dbbe9a7925ff795285619a1bd6538
  pip3 install .
  cd ..

  cp nest/Estimator/arch_configs/area_files/*.csv $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-table-based-plug-ins/set_of_table_templates/data/.
  cp -r nest/Estimator/arch_configs/area_files/tablePluginData $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
  cp -r accelergy-cacti-plug-in/cacti $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in


  ###### astra-sim related installation
  git lfs install
  git lfs pull

  cd third_party_for_nest
  tar -xzf astra-sim.tar.gz
  tar -xzf chakra.tar.gz
  tar -xzf openmpi*.tar.gz
  cd ..

  cd nest/Solver/device_placement/

  g++ device_placement.cpp -o device_placement \
    -fno-use-linker-plugin \
    -Wl,-rpath,$CONDA_PREFIX/lib \
    -lpython3.10 -lpthread -ldl -lutil -lrt

  
  # g++ device_placement.cpp -o device_placement

  cd ../../..
  
fi


if [ "$APEX_ONLY" = true ]; then
###### continue apex installation (always runs, or exclusively with --apex-only)
  ###### start apex installation 
  git clone https://github.com/NVIDIA/apex
  cd apex
  export CXX=$(which g++)
  export CC=$(which gcc)
  APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
  cd ..
fi