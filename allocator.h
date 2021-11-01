#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <sys/time.h>
#include <unordered_map>

namespace pytorch_malloc {

class Allocator {
 public:
  Allocator();
  ~Allocator();

  struct PtrInfo {
    size_t addr;
    size_t psize;
  };

  static Allocator *Instance() {
    static Allocator *allocator = new Allocator();
    return allocator;
  }

  cudaError_t malloc(void **devPtr, size_t size);
  cudaError_t free(void *devPtr);

 private:
  std::unordered_map<size_t, PtrInfo> addr2info_;
  struct timeval start_time_;
};

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H
