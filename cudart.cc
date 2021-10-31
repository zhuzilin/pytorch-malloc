#include <cuda_runtime_api.h>

#include "allocator.h"

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  pytorch_malloc::Allocator allocator = pytorch_malloc::Allocator::Instance();
  allocator.malloc(devPtr, size);
  return cudaError_t::cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  pytorch_malloc::Allocator allocator = pytorch_malloc::Allocator::Instance();
  allocator.free(devPtr);
  return cudaError_t::cudaSuccess;
}

}
