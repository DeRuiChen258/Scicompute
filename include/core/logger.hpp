#pragma once
/**
 * @file logger.hpp
 * @brief 简单日志系统
 */

#include <iostream>
#include <sstream>
#include <string>
#include <memory>

namespace sci {
namespace core {

// 日志级别
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

// 日志器
class Logger {
public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    void set_level(LogLevel level) { level_ = level; }

    void log(LogLevel level, const std::string& message) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        
        const char* prefix = "";
        switch (level) {
            case LogLevel::DEBUG: prefix = "[DEBUG]"; break;
            case LogLevel::INFO: prefix = "[INFO]"; break;
            case LogLevel::WARNING: prefix = "[WARNING]"; break;
            case LogLevel::ERROR: prefix = "[ERROR]"; break;
            case LogLevel::FATAL: prefix = "[FATAL]"; break;
        }

        std::cout << prefix << " " << message << std::endl;
    }

private:
    Logger() : level_(LogLevel::INFO) {}

    LogLevel level_;
    std::mutex mutex_;
};

} // namespace core
} // namespace sci

// 便捷宏
#define LOG_DEBUG(msg) sci::core::Logger::Instance().log(sci::core::LogLevel::DEBUG, msg)
#define LOG_INFO(msg) sci::core::Logger::Instance().log(sci::core::LogLevel::INFO, msg)
#define LOG_WARNING(msg) sci::core::Logger::Instance().log(sci::core::LogLevel::WARNING, msg)
#define LOG_ERROR(msg) sci::core::Logger::Instance().log(sci::core::LogLevel::ERROR, msg)
#define LOG_FATAL(msg) sci::core::Logger::Instance().log(sci::core::LogLevel::FATAL, msg)

// 格式化日志
#define LOG_INFO_F(fmt, ...) do { \
    char buf[256]; \
    snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
    LOG_INFO(buf); \
} while(0)

#define LOG_ERROR_F(fmt, ...) do { \
    char buf[256]; \
    snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
    LOG_ERROR(buf); \
} while(0)
