#!/bin/bash

# Paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # /scripts/tpu_fattree
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"                       # /scripts
ROOT_DIR="$(dirname "$SCRIPTS_DIR")"                         # project root
OUT_DIR="$SCRIPT_DIR/out"
mkdir -p "$OUT_DIR"

cd "$SCRIPT_DIR/../.."

EXEC_TYPE="run_solver"
MBS=1
HBM_SIZE=64
SETUP_NAME="tpuv4_fatTree"

get_devices_per_level() {
    case $1 in
        64)   echo "8 4 2"  ;;
        128)  echo "8 4 4"  ;;
        256)  echo "8 4 8"  ;;
        512)  echo "8 4 16" ;;
        1024) echo "8 4 32" ;;
        *)    echo "ERROR: Unknown num_accelerators $1"; exit 1 ;;
    esac
}

get_model_config() {
    local model=$1
    case $model in
        bert)
            MODEL_NAME="megatronbert"
            SEQ_LEN=512
            MAX_TMP_WIDTH=8
            ACTIVATION_RECOMPUTATION=""
            EXP_PARALLEL_DEGREE=""
            ;;
        llama2)
            MODEL_NAME="llama2"
            SEQ_LEN=4096
            MAX_TMP_WIDTH=""
            ACTIVATION_RECOMPUTATION="--activation_recomputation"
            EXP_PARALLEL_DEGREE=""
            ;;
        llama3)
            MODEL_NAME="llama3"
            SEQ_LEN=4096
            MAX_TMP_WIDTH=""
            ACTIVATION_RECOMPUTATION="--activation_recomputation"
            EXP_PARALLEL_DEGREE=""
            ;;
        gpt3)
            MODEL_NAME="megatrongpt3"
            SEQ_LEN=2048
            MAX_TMP_WIDTH=8
            ACTIVATION_RECOMPUTATION="--activation_recomputation"
            EXP_PARALLEL_DEGREE=""
            ;;
        mixtral)
            MODEL_NAME="mixtral"
            SEQ_LEN=4096
            MAX_TMP_WIDTH=""
            ACTIVATION_RECOMPUTATION="--activation_recomputation"
            EXP_PARALLEL_DEGREE="--exp_parallel_degree 4"
            ;;
        *)
            echo "ERROR: Unknown model '$model'. Valid options: bert, llama2, llama3, gpt3, mixtral"
            exit 1
            ;;
    esac
}

ALL_MODELS=(bert llama2 llama3 gpt3 mixtral)

# $1 = optional model name, $2 = optional accelerator count
if [ -n "$1" ]; then
    MODELS=($1)
else
    MODELS=("${ALL_MODELS[@]}")
fi

if [ -n "$2" ]; then
    ACC_COUNTS=($2)
else
    ACC_COUNTS=(64 128 256 512 1024)
fi

TOTAL_START=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    get_model_config $MODEL

    # Per-model output directory and CSV
    MODEL_OUT_DIR="$OUT_DIR/${MODEL}_mbs${MBS}"
    mkdir -p "$MODEL_OUT_DIR"
    CSV_FILE="$MODEL_OUT_DIR/${MODEL}_mbs${MBS}.csv"

    MODEL_START=$(date +%s)

    for NUM_ACC in "${ACC_COUNTS[@]}"; do
        DEVICES_PER_LEVEL=$(get_devices_per_level $NUM_ACC)
        echo "Running experiment: model=$MODEL, num_accelerators=$NUM_ACC, devices_per_level=$DEVICES_PER_LEVEL ..."

        OUTPUT_LOG="$MODEL_OUT_DIR/${MODEL}_mbs${MBS}_output_${NUM_ACC}.log"

        EXTRA_FLAGS=""
        [ -n "$MAX_TMP_WIDTH" ]            && EXTRA_FLAGS="$EXTRA_FLAGS --max_tmp_width $MAX_TMP_WIDTH"
        [ -n "$ACTIVATION_RECOMPUTATION" ] && EXTRA_FLAGS="$EXTRA_FLAGS $ACTIVATION_RECOMPUTATION"
        [ -n "$EXP_PARALLEL_DEGREE" ]      && EXTRA_FLAGS="$EXTRA_FLAGS $EXP_PARALLEL_DEGREE"

        RUN_START=$(date +%s)

        python3 nest.py \
            --model_names $MODEL_NAME \
            --exec_type $EXEC_TYPE \
            --micro_batch_size $MBS \
            --sequence_length $SEQ_LEN \
            --hbm_size $HBM_SIZE \
            --num_accelerators $NUM_ACC \
            --setup_name "$SETUP_NAME" \
            --devices_per_level $DEVICES_PER_LEVEL \
            --use_ilp \
            $EXTRA_FLAGS \
            2>&1 | tee "$OUTPUT_LOG"

        RUN_END=$(date +%s)
        RUNTIME=$((RUN_END - RUN_START))

        echo "Parsing results for model=$MODEL, num_accelerators=$NUM_ACC ..."
        python3 "$SCRIPTS_DIR/parse_results.py" \
            --log_file "$OUTPUT_LOG" \
            --num_accelerators $NUM_ACC \
            --runtime $RUNTIME \
            --csv_file "$CSV_FILE"

        echo "Done for model=$MODEL, num_accelerators=$NUM_ACC (runtime: ${RUNTIME}s)"
        echo "-------------------------------------------"
    done

    MODEL_END=$(date +%s)
    MODEL_RUNTIME=$((MODEL_END - MODEL_START))
    echo "Finished all runs for $MODEL (total model runtime: ${MODEL_RUNTIME}s)"
    echo "==========================================="
done

TOTAL_END=$(date +%s)
TOTAL_RUNTIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "All experiments complete."
echo ""

# Print summary table
echo "Generating summary and plots..."
cd $SCRIPTS_DIR
for MODEL in "${MODELS[@]}"; do
    python3 "plot_results.py" \
        --model $MODEL \
        --mbs $MBS\
        --num_devices "${ACC_COUNTS[@]}" \
        --setup_name tpuv4_fatTree \
        --out_dir "$SCRIPT_DIR/plots" 
        
    echo ""
done
cd tpuv4_fatTree
echo ""
echo "Total runtime: ${TOTAL_RUNTIME}s"