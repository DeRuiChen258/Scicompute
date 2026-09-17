#!/bin/bash
# SciComputeInfra Build Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=========================================="
echo "SciComputeInfra Build Script"
echo "=========================================="

# Default options
ENABLE_CUDA=${ENABLE_CUDA:-ON}
ENABLE_TESTS=${ENABLE_TESTS:-ON}
ENABLE_BENCHMARKS=${ENABLE_BENCHMARKS:-ON}
BUILD_TYPE=${BUILD_TYPE:-Release}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE=Debug
            shift
            ;;
        --release)
            BUILD_TYPE=Release
            shift
            ;;
        --no-cuda)
            ENABLE_CUDA=OFF
            shift
            ;;
        --no-tests)
            ENABLE_TESTS=OFF
            shift
            ;;
        --no-benchmarks)
            ENABLE_BENCHMARKS=OFF
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [ -n "$CLEAN" ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring CMake..."
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DSCI_ENABLE_CUDA="$ENABLE_CUDA" \
    -DSCI_ENABLE_OPENMP=ON \
    -DSCI_BUILD_TESTS="$ENABLE_TESTS" \
    -DSCI_BUILD_BENCHMARKS="$ENABLE_BENCHMARKS"

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="

# Run tests if enabled
if [ "$ENABLE_TESTS" = "ON" ]; then
    echo ""
    echo "Running tests..."
    ctest --output-on-failure
fi

echo ""
echo "Build artifacts are in: $BUILD_DIR"
echo ""
echo "To run the example:"
echo "  $BUILD_DIR/examples/simple_tensor"
