#!/usr/bin/env python3

import torch
import time
import ctypes
import os

dll =  ctypes.PyDLL("./libcudart.so")
# use dll to call offload all
def offload_all():
    dll.offloadAll()

def main():
    print("Testing memory allocation with offload functionality")
    print("=" * 60)
    
    # 创建一些GPU tensor进行测试
    tensors = []
    
    # 分配一些内存
    for i in range(3):
        print(f"\nAllocating tensor {i}")
        tensor = torch.randn(1024, 1024, device='cuda:0')  # 约4MB每个tensor
        tensors.append(tensor)
        print(f"Tensor {i} shape: {tensor.shape}")
    
    # 等待一下让用户看到分配信息
    time.sleep(1)
    
    # 这里可以手动调用offload功能
    # 注意：实际使用时需要通过C++接口或者JIT方式调用
    print("\n" + "="*60)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dll.pause()
    print("="*60)
    dll.resume()
    
    # 验证tensor仍然可以使用
    print("\nTesting tensor computation...")
    result = tensors[0] @ tensors[1]
    print(f"Matrix multiplication result shape: {result.shape}")
    
    # 释放内存
    print("\nFreeing tensors...")
    for i, tensor in enumerate(tensors):
        print(f"Deleting tensor {i}")
        del tensor
    
    # 强制垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
