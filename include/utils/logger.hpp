#pragma once

#include "../core/common.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>

namespace sci {
namespace logger {

// ============================================================================
// Logger
// ============================================================================
class Logger {
public:
    enum class Level {
        kTrace = 0,
        kDebug = 1,
        kInfo = 2,
        kWarn = 3,
        kError = 4,
        kCritical = 5,
    };
    
    static Logger& Instance();
    
    // Configuration
    void init(const std::string& name = "sci");
    void set_level(Level level);
    void set_pattern(const std::string& pattern);
    void add_file_sink(const std::string& path, size_t max_size_mb = 10, size_t max_files = 3);
    
    // Get logger
    std::shared_ptr<spdlog::logger> get() { return logger_; }
    
    // Convenience logging
    template<typename... Args>
    void trace(Args&&... args) { logger_->trace(std::forward<Args>(args)...); }
    
    template<typename... Args>
    void debug(Args&&... args) { logger_->debug(std::forward<Args>(args)...); }
    
    template<typename... Args>
    void info(Args&&... args) { logger_->info(std::forward<Args>(args)...); }
    
    template<typename... Args>
    void warn(Args&&... args) { logger_->warn(std::forward<Args>(args)...); }
    
    template<typename... Args>
    void error(Args&&... args) { logger_->error(std::forward<Args>(args)...); }
    
    template<typename... Args>
    void critical(Args&&... args) { logger_->critical(std::forward<Args>(args)...); }
    
private:
    Logger() = default;
    
    std::shared_ptr<spdlog::logger> logger_;
};

// Convenience macros
#define SCI_LOG_TRACE(...) ::sci::logger::Logger::Instance().trace(__VA_ARGS__)
#define SCI_LOG_DEBUG(...) ::sci::logger::Logger::Instance().debug(__VA_ARGS__)
#define SCI_LOG_INFO(...) ::sci::logger::Logger::Instance().info(__VA_ARGS__)
#define SCI_LOG_WARN(...) ::sci::logger::Logger::Instance().warn(__VA_ARGS__)
#define SCI_LOG_ERROR(...) ::sci::logger::Logger::Instance().error(__VA_ARGS__)
#define SCI_LOG_CRITICAL(...) ::sci::logger::Logger::Instance().critical(__VA_ARGS__)

} // namespace logger
} // namespace sci
