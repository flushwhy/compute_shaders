#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if ! command -v glslangValidator >/dev/null 2>&1; then
    echo "glslangValidator was not found. Install the Vulkan SDK first." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}"
shopt -s nullglob
shaders=("${SCRIPT_DIR}"/*.comp)
if (( ${#shaders[@]} == 0 )); then
    echo "No Vulkan compute shaders found in ${SCRIPT_DIR}."
    exit 0
fi

for shader in "${shaders[@]}"; do
    name="$(basename "${shader}" .comp)"
    echo "Compiling ${name}..."
    glslangValidator -V "${shader}" -o "${BUILD_DIR}/${name}.spv"
done
