#!/bin/bash
# Build wrapper script that automatically captures compilation output
# Usage: ./scripts/build_with_logging.sh [build_dir]

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default build directory
BUILD_DIR="${1:-${PROJECT_ROOT}/build}"

# Ensure results directory exists
mkdir -p "${PROJECT_ROOT}/results"

# Build and capture output
echo "Building project and capturing output to results/compilation.log..."
echo "Build directory: ${BUILD_DIR}"
echo ""

# Run cmake configure if needed
if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "Running cmake configure..."
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

# Build and capture all output
cmake --build "${BUILD_DIR}" 2>&1 | tee "${PROJECT_ROOT}/results/compilation.log"

BUILD_STATUS=${PIPESTATUS[0]}

if [ ${BUILD_STATUS} -eq 0 ]; then
    echo ""
    echo "Build successful! Output saved to results/compilation.log"
    echo ""
    echo "Results directory contents:"
    ls -lh "${PROJECT_ROOT}/results/" 2>/dev/null || echo "  (results directory is empty)"
else
    echo ""
    echo "Build failed with exit code ${BUILD_STATUS}"
    echo "Check results/compilation.log for details"
    exit ${BUILD_STATUS}
fi

