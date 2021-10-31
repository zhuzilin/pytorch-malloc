#include <stdio.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <iostream>

#include "allocator.h"

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

namespace pytorch_malloc {

Allocator::Allocator() {
  std::cout << "[Allocator] create allocator\n";
  size_t free, total;
  cuda_mem_get_info(&free, &total);
  std::cout << "[Allocator] free mem: " << free << " B, total mem: " << total << " B.\n";
  cudaError_t err = cuda_malloc(&devPtr_, free);
  // cudaMemGetInfo may give a free mem that is not accurate enough
  while (err != cudaSuccess) {
    print_last_error();
    free -= (size_t)1024 * 1024;
    err = cuda_malloc(&devPtr_, free);
  }
  std::cout << "free size: " << free << std::endl;
}

Allocator::~Allocator() {
  std::cout << "[Allocator] delete allocator\n";
  cuda_free(devPtr_);
}

void Allocator::malloc(void **devPtr, size_t size) {
  std::cout << "[Allocator] malloc: " << size << " B.\n";
  *devPtr = devPtr_;
}


void Allocator::free(void *devPtr) {
  std::cout << "[Allocator] free.\n";
}

}  // end pytorch_malloc
