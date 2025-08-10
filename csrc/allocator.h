#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <sys/time.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace pytorch_malloc {

// Logging system
enum LogLevel {
  LOG_DEBUG = 0,
  LOG_INFO = 1,
  LOG_WARN = 2,
  LOG_ERROR = 3
};

class Logger {
public:
  static Logger& getInstance() {
    static Logger instance;
    return instance;
  }
  
  void setLogLevel(LogLevel level) { log_level_ = level; }
  LogLevel getLogLevel() const { return log_level_; }
  
  template<typename... Args>
  void log(LogLevel level, Args&&... args) {
    if (level >= log_level_) {
      std::cout << getLogLevelString(level) << " ";
      (std::cout << ... << args);
      std::cout << std::endl;
    }
  }
  
private:
  Logger() : log_level_(LOG_WARN) {}  // Default to INFO level
  LogLevel log_level_;
  
  const char* getLogLevelString(LogLevel level) {
    switch (level) {
      case LOG_DEBUG: return "[DEBUG]";
      case LOG_INFO:  return "[INFO]";
      case LOG_WARN:  return "[WARN]";
      case LOG_ERROR: return "[ERROR]";
      default:        return "[UNKNOWN]";
    }
  }
};

#define LOG_DEBUG(...) Logger::getInstance().log(LOG_DEBUG, __VA_ARGS__)
#define LOG_INFO(...) Logger::getInstance().log(LOG_INFO, __VA_ARGS__)  
#define LOG_WARN(...) Logger::getInstance().log(LOG_WARN, __VA_ARGS__)
#define LOG_ERROR(...) Logger::getInstance().log(LOG_ERROR, __VA_ARGS__)

class Allocator {
 public:
  Allocator();
  ~Allocator();

  struct PtrInfo {
    CUdeviceptr virtual_addr;  // Virtual memory address (guaranteed stable)
    size_t size;               // Original requested size
    size_t aligned_size;       // Aligned size for virtual memory
    void* cpu_ptr;            // CPU pinned memory for offload (nullptr if not offloaded)
    bool is_offloaded;        // true if data is currently in CPU memory
    CUmemGenericAllocationHandle mem_handle;  // Physical memory handle
    struct timeval alloc_time;  // allocation timestamp
  };

  static Allocator *Instance() {
    static Allocator *allocator = new Allocator();
    return allocator;
  }

  cudaError_t malloc(void **devPtr, size_t size);
  cudaError_t free(void *devPtr);
  
  // Offload functionality
  cudaError_t offload_all();           // Offload all GPU memory to CPU pinned memory
  cudaError_t reload_all();            // Reload all CPU pinned memory back to GPU
  
  void enable() { enable_ = true; }
  void disable() { enable_ = false; }

  // Memory statistics
  void print_memory_stats();
  size_t get_total_allocated_size();
  size_t get_active_allocations_count();
  std::vector<PtrInfo> get_all_allocations();
  
  // Logging control
  void set_log_level(LogLevel level) { Logger::getInstance().setLogLevel(level); }

 private:
  bool enable_ = true;
  std::unordered_map<CUdeviceptr, PtrInfo> addr2info_;
  std::unordered_map<CUdeviceptr, size_t> disabled_addr2size_;  // For disabled allocations
  struct timeval start_time_;
  std::mutex allocator_mutex_;  // Thread safety for allocations
  
  // CUDA context and device info
  CUcontext cuda_context_;
  int device_id_;
  
  // Statistics
  size_t total_allocated_bytes_;
  size_t peak_allocated_bytes_;
  size_t total_allocations_count_;
  size_t total_frees_count_;
  
  // Helper functions
  cudaError_t allocate_cpu_pinned(void** hostPtr, size_t size);
  cudaError_t free_cpu_pinned(void* hostPtr);
  
  // Virtual memory management helpers  
  bool ensure_cuda_context();
  bool create_and_map_memory(size_t size, CUdeviceptr* virtual_addr, CUmemGenericAllocationHandle* mem_handle);
  void unmap_and_release_memory(CUdeviceptr virtual_addr, size_t aligned_size, CUmemGenericAllocationHandle mem_handle);
};

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H
