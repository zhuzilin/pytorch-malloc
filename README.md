# Custom PyTorch Memory Management

Compile with `nvcc`:

```bash
nvcc cudart.cc --compiler-options '-fPIC' -shared --cudart=none -o libcudart.so
```

Note that we need `--cudart=none` to prevent linking the static libcudart.so.

For more information about the nvcc flags: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html

Run the pytorch code with the generated shared object:

```bash
> LD_PRELOAD=./libcudart.so python3 torch_example.py
start allocate 0
cudaMalloc: 2097152
end allocate 0
start allocate 1
end allocate 1
start allocate 2
end allocate 2
```

To make pytorch allocate without the inherit caching mechanism, run with `PYTORCH_NO_CUDA_MEMORY_CACHING`:

```bash
> LD_PRELOAD=./libcudart.so PYTORCH_NO_CUDA_MEMORY_CACHING=1 python3 start start allocate 0
cudaMalloc: 64
end allocate 0
start allocate 1
cudaMalloc: 64
end allocate 1
start allocate 2
cudaMalloc: 64
end allocate 2
```
