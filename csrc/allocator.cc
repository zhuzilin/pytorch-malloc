#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlfcn.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>
#include <iomanip>

#include "allocator.h"

#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    CUresult error = condition;                                         \
    if (error != CUDA_SUCCESS) {                                        \
      const char* error_string;                                         \
      cuGetErrorString(error, &error_string);                          \
      std::cerr << "CUDA Error: " << error_string << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return false;                                                     \
    }                                                                   \
  } while (0)

/***************************************
 ********* CUDA Runtime API ************
 ***************************************/

void cuda_mem_get_info(size_t *free, size_t *total) {
  cudaError_t (*function)(size_t* free, size_t* total);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMemGetInfo");
  (*function)(free, total);
}

cudaError_t cuda_malloc(void **devPtr, size_t size) {
  cudaError_t (*function)(void **devPtr, size_t size);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMalloc");
  return (*function)(devPtr, size);
}

cudaError_t cuda_free(void *devPtr) {
  cudaError_t (*function)(void* devPtr);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaFree");
  return (*function)(devPtr);
}

cudaError_t cuda_malloc_host(void **ptr, size_t size) {
  cudaError_t (*function)(void **ptr, size_t size);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMallocHost");
  return (*function)(ptr, size);
}

cudaError_t cuda_free_host(void *ptr) {
  cudaError_t (*function)(void *ptr);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaFreeHost");
  return (*function)(ptr);
}

cudaError_t cuda_memcpy(void *dst, const void *src, size_t count, int kind) {
  cudaError_t (*function)(void *dst, const void *src, size_t count, int kind);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMemcpy");
  return (*function)(dst, src, count, kind);
}

void print_last_error() {
  cudaError_t (*function)(void);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaGetLastError");
  cudaError_t error = (*function)();

  char *(*str_function)(cudaError_t);
  *(void **)(&str_function) = dlsym(RTLD_NEXT, "cudaGetErrorString");
  char *err_str = (*str_function)(error);
  printf("[Allocator] error: %s\n", err_str);
}

size_t get_duration(struct timeval start, struct timeval end) {
  return (size_t)(end.tv_sec - start.tv_sec) * 1000000 + (size_t)(end.tv_usec - start.tv_usec);
}

namespace pytorch_malloc {

Allocator::Allocator() 
  : total_allocated_bytes_(0)
  , peak_allocated_bytes_(0) 
  , total_allocations_count_(0)
  , total_frees_count_(0) 
  , cuda_context_(nullptr)
  , device_id_(0) {
  
  LOG_INFO("[Allocator] create allocator with CUDA Virtual Memory Management");
  
  // Initialize CUDA Driver API
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(result, &error_string);
    std::cerr << "Failed to initialize CUDA Driver API: " << error_string << std::endl;
    return;
  }
  
  // Get current context and device
  if (!ensure_cuda_context()) {
    std::cerr << "Failed to ensure CUDA context" << std::endl;
    return;
  }
  
  size_t free, total;
  cuda_mem_get_info(&free, &total);
  LOG_INFO("[Allocator] free mem: ", free, " B, total mem: ", total, " B.");
  gettimeofday(&start_time_, NULL);
}

Allocator::~Allocator() {
  LOG_INFO("[Allocator] delete allocator");
  print_memory_stats();
  
  // Clean up any remaining offloaded memory
  for (auto& pair : addr2info_) {
    if (pair.second.cpu_ptr != nullptr) {
      free_cpu_pinned(pair.second.cpu_ptr);
    }
  }
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  if (size == 0) return cudaSuccess;
  
  std::lock_guard<std::mutex> lock(allocator_mutex_);

  if (!enable_) {
    LOG_INFO("[Allocator] malloc called while allocator is disabled");
    
    // Use traditional cudaMalloc but record the allocation
    cudaError_t result = cuda_malloc(devPtr, size);
    if (result == cudaSuccess) {
      CUdeviceptr addr = (CUdeviceptr)*devPtr;
      disabled_addr2size_[addr] = size;  // Record allocation size for tracking
      
      struct timeval timestamp;
      gettimeofday(&timestamp, NULL);
      LOG_INFO("[Allocator] disabled malloc(", addr, "): ", size / 1024 / 1024, " MB, time: ",
               get_duration(start_time_, timestamp), " us.");
    }
    
    return result;
  }

  // Allocate using virtual memory management
  CUdeviceptr virtual_addr;
  CUmemGenericAllocationHandle mem_handle;
  
  if (!create_and_map_memory(size, &virtual_addr, &mem_handle)) {
    return cudaErrorMemoryAllocation;
  }
  
  // Store allocation info
  PtrInfo info;
  info.virtual_addr = virtual_addr;
  info.size = size;
  
  // Calculate aligned size (same as in the example)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id_;
  
  size_t granularity;
  cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  info.aligned_size = ((size + granularity - 1) / granularity) * granularity;
  
  info.cpu_ptr = nullptr;
  info.is_offloaded = false;
  info.mem_handle = mem_handle;
  gettimeofday(&info.alloc_time, NULL);
  
  addr2info_[virtual_addr] = info;
  
  // Update statistics
  total_allocated_bytes_ += size;
  total_allocations_count_++;
  peak_allocated_bytes_ = std::max(peak_allocated_bytes_, total_allocated_bytes_);
  
  *devPtr = (void*)virtual_addr;
  
  struct timeval timestamp;
  gettimeofday(&timestamp, NULL);
  LOG_INFO("[Allocator] malloc(", virtual_addr, "): ", size / 1024 / 1024, " MB (aligned: ", info.aligned_size / 1024 / 1024, " MB), time: ",
           get_duration(start_time_, timestamp), " us. Total allocated: ", total_allocated_bytes_ / 1024 / 1024, " MB.");
            
  return cudaSuccess;
}

cudaError_t Allocator::free(void *devPtr) {
  std::lock_guard<std::mutex> lock(allocator_mutex_);

  if (!enable_) {
    LOG_INFO("[Allocator] free called for disabled allocator");

    struct timeval timestamp;
    gettimeofday(&timestamp, NULL);
    
    CUdeviceptr addr = (CUdeviceptr)devPtr;
    auto it = disabled_addr2size_.find(addr);
    if (it != disabled_addr2size_.end()) {
      size_t size = it->second;
      disabled_addr2size_.erase(it);
      LOG_INFO("[Allocator] disabled free(", addr, "): ", size / 1024 / 1024, " MB, time: ",
               get_duration(start_time_, timestamp), " us.");
    } else {
      LOG_INFO("[Allocator] free unknown disabled ptr ", addr, ", time: ",
               get_duration(start_time_, timestamp), " us.");
    }
    
    return cuda_free(devPtr);
  }
  
  struct timeval timestamp;
  gettimeofday(&timestamp, NULL);
  
  CUdeviceptr virtual_addr = (CUdeviceptr)devPtr;
  
  if (addr2info_.find(virtual_addr) == addr2info_.end()) {
    LOG_WARN("[Allocator] free unknown ptr ", virtual_addr, ", time: ",
             get_duration(start_time_, timestamp), " us.");
    return cudaSuccess;
  }
  
  PtrInfo info = addr2info_[virtual_addr];
  
  // Free CPU pinned memory if exists
  if (info.cpu_ptr != nullptr) {
    free_cpu_pinned(info.cpu_ptr);
    LOG_INFO("[Allocator] freed offloaded CPU memory for ptr ", virtual_addr);
  }
  
  // Unmap and release virtual memory
  unmap_and_release_memory(virtual_addr, info.aligned_size, info.mem_handle);
  
  // Free the virtual address space
  cuMemAddressFree(virtual_addr, info.aligned_size);
  
  // Update statistics
  total_allocated_bytes_ -= info.size;
  total_frees_count_++;
  
  if (info.size > 1024 * 1024) {
    LOG_INFO("[Allocator] free(", virtual_addr, "): ", info.size / 1024 / 1024, " MB, time: ",
             get_duration(start_time_, timestamp), " us. Total allocated: ", 
             total_allocated_bytes_ / 1024 / 1024, " MB.");
  }
  
  addr2info_.erase(virtual_addr);
  
  return cudaSuccess;
}

// Helper functions for virtual memory management
bool Allocator::ensure_cuda_context() {
  CUresult result = cuCtxGetCurrent(&cuda_context_);
  if (result != CUDA_SUCCESS || !cuda_context_) {
    // Get current device
    int device;
    cudaGetDevice(&device);
    device_id_ = device;
    
    // Create primary context
    result = cuDevicePrimaryCtxRetain(&cuda_context_, device_id_);
    if (result != CUDA_SUCCESS) {
      return false;
    }
    
    result = cuCtxSetCurrent(cuda_context_);
    if (result != CUDA_SUCCESS) {
      return false;
    }
  }
  
  // Get device id from current context
  CUdevice device;
  cuCtxGetDevice(&device);
  device_id_ = device;
  
  return true;
}

bool Allocator::create_and_map_memory(size_t size, CUdeviceptr* virtual_addr, CUmemGenericAllocationHandle* mem_handle) {
  if (!ensure_cuda_context()) {
    return false;
  }
  
  // Define memory allocation properties
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id_;
  prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
  
  // Get granularity and align size
  size_t granularity;
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t aligned_size = ((size + granularity - 1) / granularity) * granularity;
  
  // Reserve virtual address
  CUDA_CHECK(cuMemAddressReserve(virtual_addr, aligned_size, 0, 0, 0));
  
  // Create physical memory
  CUDA_CHECK(cuMemCreate(mem_handle, aligned_size, &prop, 0));
  
  // Map physical memory to virtual address
  CUDA_CHECK(cuMemMap(*virtual_addr, aligned_size, 0, *mem_handle, 0));
  
  // Set access permissions
  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = device_id_;
  access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  
  CUDA_CHECK(cuMemSetAccess(*virtual_addr, aligned_size, &access_desc, 1));
  
  return true;
}

void Allocator::unmap_and_release_memory(CUdeviceptr virtual_addr, size_t aligned_size, CUmemGenericAllocationHandle mem_handle) {
  if (!ensure_cuda_context()) {
    return;
  }
  
  cuMemUnmap(virtual_addr, aligned_size);
  cuMemRelease(mem_handle);
}

// Helper functions
cudaError_t Allocator::allocate_cpu_pinned(void** hostPtr, size_t size) {
  return cuda_malloc_host(hostPtr, size);
}

cudaError_t Allocator::free_cpu_pinned(void* hostPtr) {
  return cuda_free_host(hostPtr);
}

// Offload all GPU memory to CPU pinned memory
cudaError_t Allocator::offload_all() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  
  LOG_INFO("[Allocator] Starting offload of all GPU memory to CPU...");
  struct timeval start_offload;
  gettimeofday(&start_offload, NULL);
  
  size_t total_offloaded = 0;
  int count = 0;
  
  for (auto& pair : addr2info_) {
    PtrInfo& info = pair.second;
    if (!info.is_offloaded) {
      // Allocate CPU pinned memory
      void* cpu_ptr;
      cudaError_t error = allocate_cpu_pinned(&cpu_ptr, info.size);
      if (error != cudaSuccess) {
        LOG_ERROR("[Allocator] Failed to allocate CPU pinned memory for ptr ", 
                  info.virtual_addr, ", size: ", info.size, " B");
        continue;
      }
      
      // Copy data from GPU to CPU (virtual address is stable)
      error = cuda_memcpy(cpu_ptr, (void*)info.virtual_addr, info.size, 2);  // cudaMemcpyDeviceToHost = 2
      if (error != cudaSuccess) {
        LOG_ERROR("[Allocator] Failed to copy data from GPU to CPU for ptr ", 
                  info.virtual_addr);
        free_cpu_pinned(cpu_ptr);
        continue;
      }
      
      // Unmap physical memory (but keep virtual address reserved!)
      cuMemUnmap(info.virtual_addr, info.aligned_size);
      cuMemRelease(info.mem_handle);
      
      info.cpu_ptr = cpu_ptr;
      info.is_offloaded = true;
      total_offloaded += info.size;
      count++;
    }
  }
  
  struct timeval end_offload;
  gettimeofday(&end_offload, NULL);
  
  LOG_INFO("[Allocator] Offload completed: ", count, " allocations, ", 
           total_offloaded, " B, time: ", 
           get_duration(start_offload, end_offload), " us.");
  
  return cudaSuccess;
}

// Reload all CPU pinned memory back to GPU
cudaError_t Allocator::reload_all() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  
  LOG_INFO("[Allocator] Starting reload of all CPU memory back to GPU...");
  struct timeval start_reload;
  gettimeofday(&start_reload, NULL);
  
  size_t total_reloaded = 0;
  int count = 0;
  
  for (auto& pair : addr2info_) {
    PtrInfo& info = pair.second;
    if (info.is_offloaded && info.cpu_ptr != nullptr) {
      // Re-create physical memory and map to the SAME virtual address
      // This is the key advantage of virtual memory management!
      
      if (!ensure_cuda_context()) {
        LOG_ERROR("[Allocator] Failed to ensure CUDA context for ptr ", 
                  info.virtual_addr);
        continue;
      }
      
      // Create new physical memory
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_id_;
      prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;
      
      CUmemGenericAllocationHandle new_mem_handle;
      CUresult result = cuMemCreate(&new_mem_handle, info.aligned_size, &prop, 0);
      if (result != CUDA_SUCCESS) {
        LOG_ERROR("[Allocator] Failed to create physical memory for ptr ", 
                  info.virtual_addr, ", size: ", info.size, " B");
        continue;
      }
      
      // Map to the SAME virtual address (guaranteed to work!)
      result = cuMemMap(info.virtual_addr, info.aligned_size, 0, new_mem_handle, 0);
      if (result != CUDA_SUCCESS) {
        LOG_ERROR("[Allocator] Failed to map memory to virtual address ", 
                  info.virtual_addr);
        cuMemRelease(new_mem_handle);
        continue;
      }
      
      // Set access permissions
      CUmemAccessDesc access_desc = {};
      access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access_desc.location.id = device_id_;
      access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      
      result = cuMemSetAccess(info.virtual_addr, info.aligned_size, &access_desc, 1);
      if (result != CUDA_SUCCESS) {
        LOG_ERROR("[Allocator] Failed to set access permissions for ptr ", 
                  info.virtual_addr);
        cuMemUnmap(info.virtual_addr, info.aligned_size);
        cuMemRelease(new_mem_handle);
        continue;
      }
      
      // Copy data from CPU back to GPU (same virtual address!)
      cudaError_t error = cuda_memcpy((void*)info.virtual_addr, info.cpu_ptr, info.size, 1);  // cudaMemcpyHostToDevice = 1
      if (error != cudaSuccess) {
        LOG_ERROR("[Allocator] Failed to copy data from CPU to GPU for ptr ", 
                  info.virtual_addr);
        cuMemUnmap(info.virtual_addr, info.aligned_size);
        cuMemRelease(new_mem_handle);
        continue;
      }
      
      // Free CPU pinned memory and update info
      free_cpu_pinned(info.cpu_ptr);
      info.cpu_ptr = nullptr;
      info.is_offloaded = false;
      info.mem_handle = new_mem_handle;  // Update to new physical memory handle
      total_reloaded += info.size;
      count++;
      
      LOG_DEBUG("[Allocator] Successfully reloaded ptr ", info.virtual_addr, " (virtual address unchanged!)");
    }
  }
  
  struct timeval end_reload;
  gettimeofday(&end_reload, NULL);
  
  LOG_INFO("[Allocator] Reload completed: ", count, " allocations, ", 
           total_reloaded, " B, time: ", 
           get_duration(start_reload, end_reload), " us.");
  
  return cudaSuccess;
}

// Memory statistics
void Allocator::print_memory_stats() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  
  size_t gpu_memory = 0;
  size_t cpu_memory = 0;
  int gpu_count = 0;
  int cpu_count = 0;
  
  for (const auto& pair : addr2info_) {
    const PtrInfo& info = pair.second;
    if (info.is_offloaded) {
      cpu_memory += info.size;
      cpu_count++;
    } else {
      gpu_memory += info.size;
      gpu_count++;
    }
  }
  
  LOG_INFO("");
  LOG_INFO("[Allocator] Memory Statistics:");
  LOG_INFO("  Active allocations: ", addr2info_.size(), " (", 
           gpu_count, " GPU, ", cpu_count, " CPU)");
  LOG_INFO("  Current GPU memory: ", gpu_memory, " B");
  LOG_INFO("  Current CPU memory: ", cpu_memory, " B");
  LOG_INFO("  Total allocated: ", total_allocated_bytes_, " B");
  LOG_INFO("  Peak allocated: ", peak_allocated_bytes_, " B");
  LOG_INFO("  Total alloc calls: ", total_allocations_count_);
  LOG_INFO("  Total free calls: ", total_frees_count_);
  LOG_INFO("");
}

size_t Allocator::get_total_allocated_size() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  return total_allocated_bytes_;
}

size_t Allocator::get_active_allocations_count() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  return addr2info_.size();
}

std::vector<Allocator::PtrInfo> Allocator::get_all_allocations() {
  std::lock_guard<std::mutex> lock(allocator_mutex_);
  std::vector<PtrInfo> allocations;
  for (const auto& pair : addr2info_) {
    allocations.push_back(pair.second);
  }
  return allocations;
}

}  // end pytorch_malloc
