#pragma once

#include "../core/common.hpp"
#include "device.hpp"

namespace sci {

// ============================================================================
// Device Info
// ============================================================================
struct DeviceInfo {
    DeviceType type;
    int device_id;
    std::string name;
    std::string compute_capability;  // e.g., "8.0" for CUDA
    size_t total_memory;
    size_t free_memory;
    int num_multiprocessors;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    int max_grid_dim[3];
    int max_block_dim[3];
    bool supports_unified_memory;
    bool supports_cuda_managed_memory;
};

// Get device information
DeviceInfo GetDeviceInfo(DeviceType type, int device_id = 0);

// Query device capabilities
bool HasCUDA();
bool HasAvx();
bool HasAvx2();
bool HasAvx512();
bool HasNeon();

} // namespace sci
