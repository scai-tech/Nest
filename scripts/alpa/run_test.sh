#!/bin/bash

# Usage: ./run_test.sh --model <model> --num_devices <num_devices> --mbs <microbatch size>
# Models: bert, gpt3, llama2, llama3, mixtral
# Num devices: 64, 128, 256, 512
# MBS: 1, 2, 4, 8

set -e

# ─── Argument Parsing ────────────────────────────────────────────────────────
MODEL=""
NUM_DEVICES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2";       shift 2 ;;
        --num_devices) NUM_DEVICES="$2"; shift 2 ;;
        --mbs)        MBS="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$NUM_DEVICES" ]]; then
    echo "Usage: $0 --model <bert|gpt3|llama2|llama3|mixtral> --num_devices <64|128|256|512>"
    exit 1
fi

# Validate num_devices
if [[ "$NUM_DEVICES" != "64" && "$NUM_DEVICES" != "128" && "$NUM_DEVICES" != "256" && "$NUM_DEVICES" != "512" ]]; then
    echo "Error: num_devices must be one of: 64, 128, 256, 512"
    exit 1
fi

# ─── Constants ───────────────────────────────────────────────────────────────
NUM_MICROBATCH=$((4096/MBS))  # Adjust microbatch size based on number of devices
ALPA_DIR="alpa"
LOG_DIR="results/logs"
PARSED_DIR="results/parsed_results"
mkdir -p "$LOG_DIR" "$PARSED_DIR"

# ─── Model Configuration ─────────────────────────────────────────────────────
case "$MODEL" in
    bert)
        ACTUAL_NUM_LAYERS=24
        NUM_LAYERS=24
        EFFECTIVE_DEVICES=$NUM_DEVICES
        ;;
    llama2)
        ACTUAL_NUM_LAYERS=32
        NUM_LAYERS=32
        if [[ "$NUM_DEVICES" == "256" ]]; then
            NUM_LAYERS=8
            EFFECTIVE_DEVICES=$(( NUM_DEVICES / 4 ))
        elif [[ "$NUM_DEVICES" == "512" ]]; then
            NUM_LAYERS=16
            EFFECTIVE_DEVICES=$(( NUM_DEVICES / 2 ))
        else
            EFFECTIVE_DEVICES=$NUM_DEVICES
        fi
        ;;
    llama3)
        ACTUAL_NUM_LAYERS=80
        NUM_LAYERS=20
        EFFECTIVE_DEVICES=$(( NUM_DEVICES / 4 ))
        ;;
    gpt3)
        ACTUAL_NUM_LAYERS=96
        NUM_LAYERS=12
        EFFECTIVE_DEVICES=$(( NUM_DEVICES / 8 ))
        ;;
    mixtral)
        ACTUAL_NUM_LAYERS=32
        NUM_LAYERS=4
        EFFECTIVE_DEVICES=$(( NUM_DEVICES / 8 ))
        ;;
    *)
        echo "Error: Unknown model '$MODEL'. Choose from: bert, gpt3, llama2, llama3, mixtral"
        exit 1
        ;;
esac

NUM_NODES=$(( EFFECTIVE_DEVICES / 8 ))

# ─── Run ─────────────────────────────────────────────────────────────────────
SCRIPT="$ALPA_DIR/pipeshard_parallelism_${MODEL}.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${MODEL}_devices${NUM_DEVICES}_${TIMESTAMP}.log"

echo "============================================================"
echo "  Model         : $MODEL"
echo "  Num devices   : $NUM_DEVICES "
echo "  Num nodes     : $NUM_NODES"
echo "  Num layers    : $ACTUAL_NUM_LAYERS "
echo "  Num microbatch: $NUM_MICROBATCH"
echo "  Script        : $SCRIPT"
echo "  Log           : $LOG_FILE"
echo "============================================================"

python "$SCRIPT" \
    --num_nodes "$NUM_NODES" \
    --num_layers "$NUM_LAYERS" \
    --num_microbatch "$NUM_MICROBATCH" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Run complete. Log saved to: $LOG_FILE"

echo "Parsing Results: $LOG_FILE"
python parse_alpa_log.py \
    --log "$LOG_FILE" \
    --model "$MODEL" \
    --num_devices "$NUM_DEVICES" \
    --mbs "$MBS"
echo ""