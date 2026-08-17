#!/bin/bash
# SciComputeInfra Benchmark Runner Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

echo "=========================================="
echo "SciComputeInfra Benchmark Runner"
echo "=========================================="

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Running build first..."
    "$SCRIPT_DIR/build.sh"
fi

BENCHMARK_DIR="$BUILD_DIR"
OUTPUT_DIR="${PROJECT_DIR}/benchmark_results"
mkdir -p "$OUTPUT_DIR"

# Run elementwise benchmark
if [ -f "$BENCHMARK_DIR/benchmarks/bench_elementwise" ]; then
    echo ""
    echo "Running Elementwise Benchmarks..."
    "$BENCHMARK_DIR/benchmarks/bench_elementwise" --benchmark_format=json --benchmark_out="${OUTPUT_DIR}/elementwise.json"
fi

# Run reduction benchmark
if [ -f "$BENCHMARK_DIR/benchmarks/bench_reduction" ]; then
    echo ""
    echo "Running Reduction Benchmarks..."
    "$BENCHMARK_DIR/benchmarks/bench_reduction" --benchmark_format=json --benchmark_out="${OUTPUT_DIR}/reduction.json"
fi

# Run softmax benchmark
if [ -f "$BENCHMARK_DIR/benchmarks/bench_softmax" ]; then
    echo ""
    echo "Running Softmax Benchmarks..."
    "$BENCHMARK_DIR/benchmarks/bench_softmax" --benchmark_format=json --benchmark_out="${OUTPUT_DIR}/softmax.json"
fi

echo ""
echo "=========================================="
echo "Benchmarks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
