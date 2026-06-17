#!/bin/bash

# Step 2: Compute speaker-level embeddings by averaging utterance embeddings
# Reads utterance embeddings from step 1 output, groups by speaker,
# and computes average embedding per speaker using multiprocessing.

set -e

# cd to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== Configuration =====
UTTERANCES_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_w2vbert/embeddings_utterances"
SPEAKERS_DIR="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130/embeddings_w2vbert/embeddings_speakers"
MIN_UTTERANCES=1
NUM_PROCESSES=$(nproc)
CHUNK_SIZE=10
SKIP_EXISTING=true
EXCLUDE_VOICEPRINT_PREFIX=""  # set to "voiceprint" to exclude files starting with that prefix
EXCLUDE_CLONE_PATTERN=""       # set to glob pattern to exclude, e.g. "*_clone_text_*"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Speaker Embedding Computation (Multi-process) ===${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  \xF0\x9F\x93\x82 Utterances: ${UTTERANCES_DIR}"
echo -e "  \xF0\x9F\x93\x81 Speakers output: ${SPEAKERS_DIR}"
echo -e "  \xF0\x9F\x94\xA2 Min utterances: ${MIN_UTTERANCES}"
echo -e "  \xE2\x9A\xA1 Processes: ${NUM_PROCESSES}"
echo -e "  \xF0\x9F\x93\xA6 Chunk size: ${CHUNK_SIZE}"
echo -e "  \xE2\x8F\xAD Skip existing: ${SKIP_EXISTING}"
if [ -n "$EXCLUDE_VOICEPRINT_PREFIX" ]; then
    echo -e "  \xF0\x9F\x9A\xAB Exclude prefix: ${EXCLUDE_VOICEPRINT_PREFIX}"
fi
if [ -n "$EXCLUDE_CLONE_PATTERN" ]; then
    echo -e "  \xF0\x9F\x97\x91 Exclude pattern: ${EXCLUDE_CLONE_PATTERN}"
fi
echo -e "${BLUE}===============================================${NC}"

# Validate inputs
if [ ! -d "$UTTERANCES_DIR" ]; then
    echo -e "${RED}\xE2\x9D\x8C Error: Utterances directory not found: $UTTERANCES_DIR${NC}"
    echo -e "${YELLOW}  Hint: Run step1 first to extract utterance embeddings${NC}"
    exit 1
fi

mkdir -p "$SPEAKERS_DIR"

PYTHON_SCRIPT="local/compute_speaker_embeddings.py"

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
if [ -d "$UTTERANCES_DIR" ]; then
    total_utts=$(find "$UTTERANCES_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    total_spks=$(find "$UTTERANCES_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x8E\xA4 Total utterance pkl files: ${total_utts}"
    echo -e "  \xF0\x9F\x91\xA5 Total speaker dirs: ${total_spks}"

    # Show dataset breakdown
    echo -e "  \xF0\x9F\x93\x82 Dataset breakdown:"
    for ds_dir in "$UTTERANCES_DIR"/*; do
        if [ -d "$ds_dir" ]; then
            ds_name=$(basename "$ds_dir")
            ds_count=$(find "$ds_dir" -name "*.pkl" 2>/dev/null | wc -l)
            echo -e "    ${ds_name}: ${ds_count} utterances"
        fi
    done | head -10
fi

if [ -d "$SPEAKERS_DIR" ]; then
    existing=$(find "$SPEAKERS_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "  \xE2\x9C\x85 Existing speaker embeddings: ${existing}"
fi

# Build command
echo -e "${GREEN}\xF0\x9F\x9A\x80 Starting speaker embedding computation...${NC}"
echo -e "${GREEN}\xE2\x8F\xB0 Start time: $(date)${NC}"

START_TIME=$(date +%s)

CMD_ARGS=(
    --utterances_dir "$UTTERANCES_DIR"
    --speakers_dir "$SPEAKERS_DIR"
    --min_utterances "$MIN_UTTERANCES"
    --num_processes "$NUM_PROCESSES"
    --chunk_size "$CHUNK_SIZE"
)

if [ "$SKIP_EXISTING" = true ]; then
    CMD_ARGS+=(--skip_existing)
fi

if [ -n "$EXCLUDE_VOICEPRINT_PREFIX" ]; then
    CMD_ARGS+=(--exclude_filename_prefix "$EXCLUDE_VOICEPRINT_PREFIX")
fi

if [ -n "$EXCLUDE_CLONE_PATTERN" ]; then
    CMD_ARGS+=(--exclude_filename_pattern "$EXCLUDE_CLONE_PATTERN")
fi

python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))

echo -e "${GREEN}\xE2\x9C\x85 Speaker embedding computation completed!${NC}"
echo -e "${GREEN}\xE2\x8F\xB0 End time: $(date)${NC}"
echo -e "${GREEN}\xE2\x8F\xB1 Total time: ${EXECUTION_TIME}s ($(printf '%02d:%02d:%02d' $((EXECUTION_TIME/3600)) $((EXECUTION_TIME%3600/60)) $((EXECUTION_TIME%60))))${NC}"

# Post-run stats
echo -e "${BLUE}\xF0\x9F\x93\x88 Final statistics:${NC}"
if [ -d "$SPEAKERS_DIR" ]; then
    final_spks=$(find "$SPEAKERS_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "  \xF0\x9F\x91\xA5 Total speaker embeddings: ${final_spks}"

    echo -e "  \xF0\x9F\x93\x82 Dataset breakdown:"
    for ds_dir in "$SPEAKERS_DIR"/*; do
        if [ -d "$ds_dir" ]; then
            ds_name=$(basename "$ds_dir")
            spk_count=$(find "$ds_dir" -name "*.pkl" 2>/dev/null | wc -l)
            echo -e "    ${ds_name}: ${spk_count} speakers"
        fi
    done | head -10

    if [ $EXECUTION_TIME -gt 0 ] && [ $final_spks -gt 0 ]; then
        rate=$(echo "scale=2; $final_spks / $EXECUTION_TIME" | bc -l)
        echo -e "  \xF0\x9F\x9A\x80 Rate: ${rate} speakers/second"
    fi
fi

# Disk usage
echo -e "${BLUE}\xF0\x9F\x92\xBE Disk usage:${NC}"
spk_size=$(du -sh "$SPEAKERS_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo -e "  Speakers dir: ${spk_size}"

echo -e "${GREEN}\xF0\x9F\x8E\x89 Done.${NC}"
