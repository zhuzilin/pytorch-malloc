#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <iostream>

extern "C" {

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
  cudaError_t (*function)(void **devPtr, size_t size);
  *(void **)(&function) = dlsym(RTLD_NEXT, "cudaMalloc");
  std::cout << "cudaMalloc: " << size << " bytes." << std::endl;
  return (*function)(devPtr, size);
}

}
