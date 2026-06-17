#!/bin/bash

# Step 1: Extract W2V-BERT speaker embeddings from audio files
# Supports multi-GPU parallel processing via file sharding.

set -e

# cd to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== Configuration =====
BASE="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
DATA_ROOT="${BASE}/audio"
CHECKPOINT="/root/code/github_repos/w2v-BERT-2.0_SV-fork/models/model_lmft_0.14.pth"
TRAIN_YAML="/root/code/github_repos/w2v-BERT-2.0_SV-fork/models/train.yaml"
OUTPUT_DIR="${BASE}/embeddings_w2vbert/embeddings_utterances"
NUM_GPUS=4                 # number of GPUs to use
PROCS_PER_GPU=4             # processes per GPU (total processes = NUM_GPUS * PROCS_PER_GPU)
NUM_WORKERS=2               # I/O workers per process
SKIP_EXISTING=true
RANDOM_SHUFFLE=true
RANDOM_SEED=42
MAX_FILES=0                # 0 = all; >0 = limit total across all processes
CONDA_ENV="asv"            # conda environment name

TOTAL_PROCS=$((NUM_GPUS * PROCS_PER_GPU))
TOTAL_STRIDE=$TOTAL_PROCS

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== W2V-BERT Speaker Embedding Extraction (${NUM_GPUS} GPUs x ${PROCS_PER_GPU} procs = ${TOTAL_PROCS} total) ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Data root: ${DATA_ROOT}"
echo -e "  Checkpoint: ${CHECKPOINT}"
echo -e "  Train YAML: ${TRAIN_YAML}"
echo -e "  Output: ${OUTPUT_DIR}"
echo -e "  GPUs: ${NUM_GPUS} x ${PROCS_PER_GPU} procs each (stride=${TOTAL_STRIDE})"
echo -e "  Workers per proc: ${NUM_WORKERS}"
echo -e "  Skip existing: ${SKIP_EXISTING}"
echo -e "${BLUE}===============================================${NC}"

# Validate inputs
for req in "$CHECKPOINT" "$TRAIN_YAML"; do
    if [ ! -f "$req" ]; then
        echo -e "${RED}Error: not found: $req${NC}"
        exit 1
    fi
done
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}Error: data dir not found: $DATA_ROOT${NC}"
    exit 1
fi

PYTHON_SCRIPT="local/extract_embeddings.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Build common args
COMMON_ARGS=(
    --data_root "$DATA_ROOT"
    --checkpoint "$CHECKPOINT"
    --train_yaml "$TRAIN_YAML"
    --output_dir "$OUTPUT_DIR"
    --file_stride "$TOTAL_STRIDE"
    --num_workers "$NUM_WORKERS"
    --random_seed "$RANDOM_SEED"
)
[ "$SKIP_EXISTING" = true ] && COMMON_ARGS+=(--skip_existing)
[ "$RANDOM_SHUFFLE" = true ] && COMMON_ARGS+=(--random_shuffle)
[ "$MAX_FILES" -gt 0 ] && COMMON_ARGS+=(--max_files "$MAX_FILES")

# Activate conda and launch N workers
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"

echo -e "${GREEN}Launching ${TOTAL_PROCS} processes on ${NUM_GPUS} GPUs...${NC}"
echo -e "${GREEN}Start time: $(date)${NC}"
START_TIME=$(date +%s)

PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((p=0; p<PROCS_PER_GPU; p++)); do
        offset=$((gpu + p * NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$gpu \
        python3 "$PYTHON_SCRIPT" \
            --device "cuda:0" \
            --file_offset "$offset" \
            "${COMMON_ARGS[@]}" \
            > "/tmp/emb_w2v_gpu${gpu}_p${p}.log" 2>&1 &
        PIDS+=($!)
        echo "  GPU${gpu}.proc${p} (offset=${offset}): PID ${PIDS[-1]}"
    done
done

echo "Waiting for all ${TOTAL_PROCS} processes..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo -e "  ${GREEN}[$((i+1))/${TOTAL_PROCS}] done${NC}"
    else
        echo -e "  ${RED}[$((i+1))/${TOTAL_PROCS}] FAILED${NC}"
        FAILED=1
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "${GREEN}End time: $(date)${NC}"
echo -e "${GREEN}Wall time: ${ELAPSED}s ($(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))${NC}"

# Post-run stats
if [ -d "$OUTPUT_DIR" ]; then
    extracted=$(find "$OUTPUT_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "Total embeddings: ${extracted}"
    if [ $extracted -gt 0 ] && [ $ELAPSED -gt 0 ]; then
        rate=$(echo "scale=1; $extracted / $ELAPSED" | bc -l)
        echo -e "Effective rate: ${rate} utt/s (${TOTAL_PROCS} procs on ${NUM_GPUS} GPUs)"
    fi
fi

if [ "$FAILED" -eq 1 ]; then
    echo -e "${RED}Some workers failed. Check /tmp/emb_w2v_gpu*_p*.log${NC}"
    exit 1
fi

echo -e "${GREEN}Done.${NC}"
