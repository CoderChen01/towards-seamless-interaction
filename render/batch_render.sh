#!/bin/zsh

# ========================
# Basic configuration
# ========================
TIMAR_PRED_PATH=timar_results/timar/main_test_result_1s_n_7
GT_PATH=/data/chenjj/projects/digital-human/DualTalk/data/test

# ✅ Multiple comparison paths and labels supported
COMP_PATHS=(
    timar_results/timar/main_test_result_cfg0
    timar_results/timar/main_test_result_cfg1.5
    timar_results/timar/main_test_result_cfg2.5
    timar_results/timar/main_test_result_cfg3.5
    timar_results/timar/main_test_result_cfg4.5
    timar_results/dualtalk/main_test_result_1s_n_7
)
COMP_LABELS=(
    'TIMAR Agent CFG0'
    'TIMAR Agent CFG1.5'
    'TIMAR Agent CFG2.5'
    'TIMAR Agent CFG3.5'
    'TIMAR Agent CFG4.5'
    'DualTalk Agent'
)

OUTPUT_PATH=render_result/cfg_test_all
INTEREST_OF_SAMPLES_PATH=interest_of_sample_list_cfg.txt

# ========================
# GPU configuration
# ========================
AVAILABLE_GPUS=(0 1 2 3 4 5 6)
MAX_RETRIES=3
RETRY_SLEEP=5

# ========================
# Prepare log directory
# ========================
LOG_DIR=logs
mkdir -p "${LOG_DIR}"

# ========================
# Load sample list and split
# ========================
# Read non-empty lines as names (array)
name_list=($(grep -v '^$' "${INTEREST_OF_SAMPLES_PATH}"))

gpu_count=${#AVAILABLE_GPUS[@]}
total_samples=${#name_list[@]}
chunk_size=$(( (total_samples + gpu_count - 1) / gpu_count ))

# ========================
# Sanitize OUTPUT_PATH for use in filenames
# ========================
sanitized_output_name=${OUTPUT_PATH//\//_}

# ========================
# Function: run one GPU job with retry & GPU switching
# ========================
run_job() {
    local init_dev=$1
    shift
    local ios_list=("$@")  # array of names for this job
    local log_file="${LOG_DIR}/${sanitized_output_name}_gpu${init_dev}.txt"

    # Nothing to run
    if [[ ${#ios_list[@]} -eq 0 ]]; then
        echo "[GPU ${init_dev}] No samples assigned, skipping." >> "${log_file}"
        return 0
    fi

    local attempt=1
    local exit_code=1
    local current_dev=${init_dev}

    while (( attempt <= MAX_RETRIES )); do
        echo "[GPU ${current_dev}] Attempt ${attempt} started at $(date)" >> "${log_file}"

        # NOTE: --interest_of_sample_list is LAST, and we DO NOT insert a lone `--`
        python -u render.py \
            --timar_pred_path "${TIMAR_PRED_PATH}" \
            --gt_path "${GT_PATH}" \
            --output_path "${OUTPUT_PATH}" \
            --comp_paths "${COMP_PATHS[@]}" \
            --comp_labels "${COMP_LABELS[@]}" \
            --device "cuda:${current_dev}" \
            --batch_size 125 \
            --interest_of_sample_list "${ios_list[@]}" \
            >> "${log_file}" 2>&1

        exit_code=$?

        if [[ $exit_code -eq 0 ]]; then
            echo "[GPU ${current_dev}] Completed successfully at $(date)" >> "${log_file}"
            return 0
        else
            echo "[GPU ${current_dev}] Failed (exit ${exit_code}) at $(date)" >> "${log_file}"
            (( attempt++ ))

            # Round-robin switch to the next GPU
            local idx=1
            for ((idx=1; idx<=${#AVAILABLE_GPUS[@]}; idx++)); do
                if [[ ${AVAILABLE_GPUS[$idx]} -eq ${current_dev} ]]; then
                    break
                fi
            done
            local next_idx=$(( (idx % ${#AVAILABLE_GPUS[@]}) + 1 ))
            current_dev=${AVAILABLE_GPUS[$next_idx]}

            echo "[GPU ${init_dev}] Retrying on GPU ${current_dev} after ${RETRY_SLEEP}s..." >> "${log_file}"
            sleep "${RETRY_SLEEP}"
        fi
    done

    echo "[GPU ${init_dev}] Gave up after ${MAX_RETRIES} attempts (last GPU ${current_dev})." >> "${log_file}"
    return 1
}

# ========================
# Assign samples to GPUs and run asynchronously
# ========================
pids=()

for ((i=1; i<=gpu_count; i++)); do
    dev=${AVAILABLE_GPUS[$i]}
    start=$(( (i - 1) * chunk_size ))
    ios_list=("${name_list[@]:$start:$chunk_size}")

    # Launch background job
    run_job "${dev}" "${ios_list[@]}" &
    pids+=($!)
done

# ========================
# Wait for all background jobs to finish
# ========================
for pid in "${pids[@]}"; do
    wait "$pid"
done

# ========================
# Final message
# ========================
echo "✅ All render tasks completed (with GPU retry & fallback)."
echo "Logs saved in: ${LOG_DIR}/"