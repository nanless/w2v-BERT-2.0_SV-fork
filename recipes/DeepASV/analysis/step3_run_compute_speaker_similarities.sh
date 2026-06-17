#!/bin/bash

# Step 3: Compute speaker-to-speaker similarities
# Reads speaker embeddings from step 2 output and computes pairwise cosine
# similarities, generating analysis reports and top-K similar pairs.

set -e

# cd to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== Configuration =====
EMBEDDINGS_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_w2vbert"
UTTERANCES_SUBDIR="embeddings_utterances"
SPEAKERS_SUBDIR="embeddings_speakers"
SIMILARITIES_SUBDIR="speaker_similarity_analysis"
NUM_WORKERS=32
BATCH_SIZE=100
TOP_K=100
SKIP_SIMILARITY=false
MAX_SPEAKERS=0  # 0 = all speakers
EXCLUDE_CLONE_PATTERN=""  # e.g. "*_clone_text_*"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Speaker Similarity Computation ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  \xF0\x9F\x93\x82 Embeddings dir: ${EMBEDDINGS_DIR}"
echo -e "  \xF0\x9F\x93\x81 Utterances subdir: ${UTTERANCES_SUBDIR}"
echo -e "  \xF0\x9F\x93\x81 Speakers subdir: ${SPEAKERS_SUBDIR}"
echo -e "  \xF0\x9F\x93\x81 Similarities output: ${SIMILARITIES_SUBDIR}"
echo -e "  \xE2\x9A\xA1 Workers: ${NUM_WORKERS}"
echo -e "  \xF0\x9F\x93\xA6 Batch size: ${BATCH_SIZE}"
echo -e "  \xF0\x9F\x94\x9D Top-K: ${TOP_K}"
echo -e "  \xE2\x8F\xAD Skip similarity: ${SKIP_SIMILARITY}"
echo -e "  \xF0\x9F\x91\xA5 Max speakers: ${MAX_SPEAKERS:-all}"
if [ -n "$EXCLUDE_CLONE_PATTERN" ]; then
    echo -e "  \xF0\x9F\x97\x91 Exclude pattern: ${EXCLUDE_CLONE_PATTERN}"
fi
echo -e "${BLUE}===============================================${NC}"

# Validate inputs
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo -e "${RED}\xE2\x9D\x8C Error: Embeddings directory not found: $EMBEDDINGS_DIR${NC}"
    exit 1
fi

SPEAKERS_FULL_PATH="$EMBEDDINGS_DIR/$SPEAKERS_SUBDIR"
if [ ! -d "$SPEAKERS_FULL_PATH" ]; then
    echo -e "${RED}\xE2\x9D\x8C Error: Speakers directory not found: $SPEAKERS_FULL_PATH${NC}"
    echo -e "${YELLOW}  Hint: Run step2 first to compute speaker embeddings${NC}"
    exit 1
fi

UTTERANCES_FULL_PATH="$EMBEDDINGS_DIR/$UTTERANCES_SUBDIR"
SIMILARITIES_FULL_PATH="$EMBEDDINGS_DIR/$SIMILARITIES_SUBDIR"

PYTHON_SCRIPT="local/compute_speaker_similarities.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}\xE2\x9D\x8C Error: Python script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# System info
echo -e "${BLUE}\xF0\x9F\x92\xBB System info:${NC}"
echo -e "  CPU cores: $(nproc)"
echo -e "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo -e "  Python: $(python3 --version)"

# Pre-run stats
echo -e "${BLUE}\xF0\x9F\x93\x8A Pre-computation statistics:${NC}"
if [ -d "$UTTERANCES_FULL_PATH" ]; then
    total_utts=$(find "$UTTERANCES_FULL_PATH" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x8E\xA4 Total utterance pkl files: ${total_utts}"
fi

if [ -d "$SPEAKERS_FULL_PATH" ]; then
    total_spks=$(find "$SPEAKERS_FULL_PATH" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x91\xA5 Total speaker embeddings: ${total_spks}"

    echo -e "  \xF0\x9F\x93\x82 Dataset breakdown:"
    for ds_dir in "$SPEAKERS_FULL_PATH"/*; do
        if [ -d "$ds_dir" ]; then
            ds_name=$(basename "$ds_dir")
            spk_count=$(find "$ds_dir" -name "*.pkl" 2>/dev/null | wc -l)
            echo -e "    ${ds_name}: ${spk_count} speakers"
        fi
    done | head -10
fi

if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    existing=$(find "$SIMILARITIES_FULL_PATH" -name "*.json" 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x93\x8A Existing similarity results: ${existing} files"
fi

# Build command
echo -e "${GREEN}\xF0\x9F\x9A\x80 Starting similarity computation...${NC}"
echo -e "${GREEN}\xE2\x8F\xB0 Start time: $(date)${NC}"

START_TIME=$(date +%s)

CMD_ARGS=(
    --embeddings_dir "$EMBEDDINGS_DIR"
    --speakers_subdir "$SPEAKERS_SUBDIR"
    --similarities_output_subdir "$SIMILARITIES_SUBDIR"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCH_SIZE"
    --top_k "$TOP_K"
)

if [ "$SKIP_SIMILARITY" = true ]; then
    CMD_ARGS+=(--skip_similarity)
fi

if [ "$MAX_SPEAKERS" -gt 0 ]; then
    CMD_ARGS+=(--max_speakers "$MAX_SPEAKERS")
fi

if [ -n "$EXCLUDE_CLONE_PATTERN" ]; then
    CMD_ARGS+=(--exclude_filename_pattern "$EXCLUDE_CLONE_PATTERN")
fi

# Set environment for better performance
export PYTHONIOENCODING=UTF-8

python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}\xE2\x9C\x85 Similarity computation completed!${NC}"
echo -e "${GREEN}\xE2\x8F\xB0 End time: $(date)${NC}"
echo -e "${GREEN}\xE2\x8F\xB1 Total time: ${EXECUTION_TIME}s ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Final stats
echo -e "${BLUE}\xF0\x9F\x93\x88 Final statistics:${NC}"
if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    final_results=$(find "$SIMILARITIES_FULL_PATH" -name "*.json" 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x93\x8A Total result files: ${final_results}"

    echo -e "  \xF0\x9F\x93\x84 Key output files:"
    key_files=("speaker_keys_mapping.json" "speaker_top_similarities.json" "analysis_summary.json" "extreme_similarity_pairs.json" "threshold_statistics.json")
    for f in "${key_files[@]}"; do
        if [ -f "$SIMILARITIES_FULL_PATH/$f" ]; then
            fsize=$(du -h "$SIMILARITIES_FULL_PATH/$f" | cut -f1)
            echo -e "    \xE2\x9C\x85 $f (${fsize})"
        else
            echo -e "    \xE2\x9D\x8C $f (missing)"
        fi
    done
fi

# Disk usage
echo -e "${BLUE}\xF0\x9F\x92\xBE Disk usage:${NC}"
if [ -d "$SIMILARITIES_FULL_PATH" ]; then
    sim_size=$(du -sh "$SIMILARITIES_FULL_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Similarities dir: ${sim_size}"
fi

# Next steps
echo -e "${YELLOW}\xF0\x9F\x94\x84 Next steps:${NC}"
echo -e "  \xE2\x80\xA2 Check analysis summary:  ${SIMILARITIES_FULL_PATH}/analysis_summary.json"
echo -e "  \xE2\x80\xA2 Review extreme pairs:    ${SIMILARITIES_FULL_PATH}/extreme_similarity_pairs.json"
echo -e "  \xE2\x80\xA2 Top-K per speaker:       ${SIMILARITIES_FULL_PATH}/speaker_top_similarities.json"
echo -e "  \xE2\x80\xA2 Threshold statistics:    ${SIMILARITIES_FULL_PATH}/threshold_statistics.json"

echo -e "${GREEN}\xF0\x9F\x8E\x89 Done. Results in: ${SIMILARITIES_FULL_PATH}${NC}"
