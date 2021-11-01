#include <stdio.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <sys/time.h>

#include <iostream>

#include "allocator.h"

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

Allocator::Allocator() {
  std::cout << "[Allocator] create allocator\n";
  size_t free, total;
  cuda_mem_get_info(&free, &total);
  std::cout << "[Allocator] free mem: " << free << " B, total mem: " << total << " B.\n";
  gettimeofday(&start_time_, NULL);
}

Allocator::~Allocator() {
  std::cout << "[Allocator] delete allocator\n";
}

cudaError_t Allocator::malloc(void **devPtr, size_t size) {
  if (size == 0) return cudaSuccess;
  cudaError_t error = cuda_malloc(devPtr, size);
  PtrInfo info;
  info.addr = (size_t)(*devPtr);
  info.psize = size;
  addr2info_[(size_t)(*devPtr)] = info;
  struct timeval timestamp;
  gettimeofday(&timestamp, NULL);
  std::cout << "[Allocator] malloc(" << info.addr << "): " << size << " B, time: "
            << get_duration(start_time_, timestamp) << " us.\n";
  return error;
}

cudaError_t Allocator::free(void *devPtr) {
  cudaError_t error = cuda_free(devPtr);
  struct timeval timestamp;
  gettimeofday(&timestamp, NULL);
  if (addr2info_.find((size_t)devPtr) == addr2info_.end()) {
    std::cout << "[Allocator] free unknown ptr, time: "
              << get_duration(start_time_, timestamp) << " us.\n";;
  } else {
    PtrInfo info = addr2info_[(size_t)devPtr];
    std::cout << "[Allocator] free(" << info.addr << "): " << info.psize << " B, time: "
              << get_duration(start_time_, timestamp) << " us.\n";;
  }
  return error;
}

}  // end pytorch_malloc
