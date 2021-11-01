#include <cuda_runtime_api.h>

#include "allocator.h"

extern "C" {

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  pytorch_malloc::Allocator *allocator = pytorch_malloc::Allocator::Instance();
  return allocator->free(devPtr);
}

}
