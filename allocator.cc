#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <iostream>

#include "allocator.h"

void cuda_mem_get_info(size_t *free, size_t *total) {
  cudaError_t (*function)(size_t* free, size_t* total);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMemGetInfo");
  function(free, total);
}

void cuda_malloc(void **devPtr, size_t size) {
  cudaError_t (*function)(void **devPtr, size_t size);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMalloc");
  (*function)(devPtr, size);
}

void cuda_free(void *devPtr) {
  cudaError_t (*function)(void* devPtr);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaFree");
  (*function)(devPtr);
}

namespace pytorch_malloc {

Allocator::Allocator() {
  size_t free, total;
  cuda_mem_get_info(&free, &total);
  std::cout << "[Allocator] free mem: " << free << "B, total mem: " << total << " B." << std::endl;
}

void Allocator::malloc(void **devPtr, size_t size) {
  std::cout << "[Allocator] malloc: " << size << " B." << std::endl;
  cuda_malloc(devPtr, size);
}


void Allocator::free(void *devPtr) {
  std::cout << "[Allocator] free." << std::endl;
  cuda_free(devPtr);
}

}  // end pytorch_malloc

