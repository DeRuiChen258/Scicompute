// ============================================================================
// SciComputeInfra - CUDA Kernel Demo
// ============================================================================

#include <iostream>
#include "core/common.hpp"
#include "device/device.hpp"
#include "cuda/kernels/elementwise.cuh"

using namespace sci;

int main() {
    std::cout << "========================================\n";
    std::cout << "SciComputeInfra - CUDA Kernel Demo\n";
    std::cout << "========================================\n\n";
    
    // Note: This demo requires CUDA-capable GPU
    // Currently running in CPU fallback mode
    
    std::cout << "CUDA support: Check SCI_HAS_CUDA macro\n";
    
    constexpr size_t N = 1024;
    
    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_out = new float[N];
    
    // Initialize
    for (size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = 2.0f;
    }
    
    std::cout << "Input a[0]: " << h_a[0] << "\n";
    std::cout << "Input b[0]: " << h_b[0] << "\n";
    
    // In a full CUDA implementation, we would:
    // 1. Allocate device memory with cudaMalloc
    // 2. Copy input data with cudaMemcpy
    // 3. Launch kernel with launch_add<float>
    // 4. Copy result back with cudaMemcpy
    // 5. Free device memory
    
    std::cout << "\nCUDA kernel would compute: out = a + b\n";
    std::cout << "Expected output[0]: " << (h_a[0] + h_b[0]) << "\n";
    
    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
    
    std::cout << "\n========================================\n";
    std::cout << "CUDA Demo completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}
