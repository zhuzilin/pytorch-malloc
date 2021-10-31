#ifndef ALLOCATOR_H
#define ALLOCATOR_H

namespace pytorch_malloc {

class Allocator {
 public:
  Allocator();

  static Allocator& Instance() {
    static Allocator *allocator = new Allocator();
    return *allocator;
  }

  void malloc(void **devPtr, size_t size);
  void free(void *devPtr);
};

}  // namespace pytorch_malloc

#endif  // ALLOCATOR_H