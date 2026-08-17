#pragma once

#include "types.hpp"
#include <string>
#include <map>
#include <memory>

namespace sci {

// ============================================================================
// Configuration
// ============================================================================
struct CoreConfig {
    bool use_sse = true;
    bool use_avx = true;
    bool use_avx512 = false;
    bool use_openmp = true;
    size_t num_threads = 0;
    bool enable_profiling = false;
    size_t alignment = kDefaultAlignment;
};

struct MemoryConfig {
    size_t host_page_size = 4096;
    size_t device_page_size = 4096;
    bool use_cuda_managed_memory = false;
    size_t memory_pool_size = 0;
};

struct DeviceConfig {
    int cuda_device_id = 0;
    bool cuda_allow_tf32 = true;
    bool cuda_use_cublas = true;
    int cuda_stream_priority_min = 0;
    int cuda_stream_priority_max = 0;
};

struct SchedulerConfig {
    size_t max_concurrent_tasks = 16;
    size_t task_queue_size = 1024;
    bool enable_work_stealing = true;
    bool enable_task_fusion = false;
};

struct Config {
    std::string name = "SciComputeInfra";
    std::string version = "0.1.0";
    CoreConfig core;
    MemoryConfig memory;
    DeviceConfig device;
    SchedulerConfig scheduler;
};

// ============================================================================
// Global Config Access
// ============================================================================
class ConfigManager {
public:
    static ConfigManager& Instance();
    
    Config& config() { return config_; }
    const Config& config() const { return config_; }
    
    void set_config(const Config& config) { config_ = config; }

private:
    ConfigManager() = default;
    Config config_;
};

inline Config& GetGlobalConfig() { return ConfigManager::Instance().config(); }

} // namespace sci
