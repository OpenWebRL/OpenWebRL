#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Usage:
#   Single iter:
#     bash scripts/run_convert_hf.sh \
#       /path/to/run/iter_0000019
#
#   Batch convert all iter_* under a run dir:
#     bash scripts/run_convert_hf.sh \
#       /path/to/run_dir
#
# Optional env vars:
#   ORIGIN_HF_DIR   Original HF model dir used to construct the correct bridge provider
#   PYTHONPATH      Defaults to /root/Megatron-LM
#
# Output layout:
#   For run dir /path/to/run_name and iter /path/to/run_name/iter_0000019
#   output dir becomes:
#     /path/to/run_name/iter_0000019/run_name_converted

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: bash scripts/run_convert_hf.sh <input_iter_dir_or_run_dir> [origin_hf_dir]"
    exit 1
fi

INPUT_PATH="${1%/}"
ORIGIN_HF_DIR="${2:-${ORIGIN_HF_DIR:-}}"

if [ ! -d "${INPUT_PATH}" ]; then
    echo "ERROR: input path not found: ${INPUT_PATH}"
    exit 1
fi

if [ ! -d "${ORIGIN_HF_DIR}" ]; then
    echo "ERROR: origin HF dir not found: ${ORIGIN_HF_DIR}"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-/root/Megatron-LM}"

find_metadata_source_iter() {
    local run_dir="$1"
    local iter_dir

    while IFS= read -r -d '' iter_dir; do
        if [ -f "${iter_dir}/.metadata" ] || [ -f "${iter_dir}/metadata.json" ]; then
            echo "${iter_dir}"
            return 0
        fi
    done < <(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' -print0 | sort -z)

    return 1
}

ensure_metadata_files() {
    local iter_dir="$1"
    local metadata_source_iter="$2"

    if [ ! -f "${iter_dir}/.metadata" ]; then
        if [ -f "${metadata_source_iter}/.metadata" ]; then
            cp "${metadata_source_iter}/.metadata" "${iter_dir}/.metadata"
            echo "Copied .metadata -> ${iter_dir}"
        else
            echo "ERROR: missing source .metadata in ${metadata_source_iter}"
            return 1
        fi
    fi

    if [ ! -f "${iter_dir}/metadata.json" ]; then
        if [ -f "${metadata_source_iter}/metadata.json" ]; then
            cp "${metadata_source_iter}/metadata.json" "${iter_dir}/metadata.json"
            echo "Copied metadata.json -> ${iter_dir}"
        else
            echo "ERROR: missing source metadata.json in ${metadata_source_iter}"
            return 1
        fi
    fi
}

cleanup_distcp_shards() {
    local iter_dir="$1"
    local shard_count

    shard_count="$(find "${iter_dir}" -maxdepth 1 -type f -name '__*.distcp' | wc -l)"
    if [ "${shard_count}" -eq 0 ]; then
        return 0
    fi

    echo "Found ${shard_count} distcp shard file(s) in ${iter_dir}; removing them after successful conversion."
    if ! find "${iter_dir}" -maxdepth 1 -type f -name '__*.distcp' -delete; then
        echo "WARNING: failed to remove one or more distcp shard files under ${iter_dir}" >&2
        return 0
    fi

    echo "Removed distcp shard files from ${iter_dir}"
}

convert_iter_dir() {
    local iter_dir="$1"
    local run_dir="$2"
    local metadata_source_iter="$3"
    local run_name iter_name converted_name output_dir

    if [ ! -f "${iter_dir}/common.pt" ]; then
        echo "Skipping ${iter_dir}: missing common.pt"
        return 0
    fi

    ensure_metadata_files "${iter_dir}" "${metadata_source_iter}"

    run_name="$(basename "${run_dir}")"
    iter_name="$(basename "${iter_dir}")"
    converted_name="${run_name}_${iter_name}_converted"
    output_dir="${iter_dir}/${converted_name}"

    if [ -d "${output_dir}" ]; then
        echo "Skipping ${iter_dir}: output already exists at ${output_dir}"
        return 0
    fi

    echo "REPO_ROOT:      ${REPO_ROOT}"
    echo "ITER_DIR:       ${iter_dir}"
    echo "OUTPUT_DIR:     ${output_dir}"
    echo "ORIGIN_HF_DIR:  ${ORIGIN_HF_DIR}"
    echo "PYTHONPATH:     ${PYTHONPATH}"

    (
        cd "${REPO_ROOT}"
        python tools/convert_torch_dist_to_hf_bridge.py \
            --input-dir "${iter_dir}" \
            --output-dir "${output_dir}" \
            --origin-hf-dir "${ORIGIN_HF_DIR}" \
            -f
    )

    if [ ! -d "${output_dir}" ]; then
        echo "ERROR: conversion finished but output dir was not created: ${output_dir}"
        return 1
    fi

    # cleanup_distcp_shards "${iter_dir}"
    echo "HF checkpoint exported to: ${output_dir}"
}

if [ -f "${INPUT_PATH}/common.pt" ]; then
    RUN_DIR="$(dirname "${INPUT_PATH}")"
    if ! METADATA_SOURCE_ITER="$(find_metadata_source_iter "${RUN_DIR}")"; then
        echo "ERROR: could not find any iter_* with .metadata or metadata.json under ${RUN_DIR}"
        exit 1
    fi
    convert_iter_dir "${INPUT_PATH}" "${RUN_DIR}" "${METADATA_SOURCE_ITER}"
    exit 0
fi

RUN_DIR="${INPUT_PATH}"
if ! METADATA_SOURCE_ITER="$(find_metadata_source_iter "${RUN_DIR}")"; then
    echo "ERROR: could not find any iter_* with .metadata or metadata.json under ${RUN_DIR}"
    exit 1
fi

echo "RUN_DIR:             ${RUN_DIR}"
echo "METADATA_SOURCE_ITER:${METADATA_SOURCE_ITER}"

found_any=0
while IFS= read -r -d '' iter_dir; do
    found_any=1
    convert_iter_dir "${iter_dir}" "${RUN_DIR}" "${METADATA_SOURCE_ITER}"
done < <(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'iter_*' -print0 | sort -z)

if [ "${found_any}" -eq 0 ]; then
    echo "ERROR: no iter_* directories found under ${RUN_DIR}"
    exit 1
fi
