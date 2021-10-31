libcudart.so:
	nvcc cudart.cc allocator.cc --compiler-options '-fPIC' -shared --cudart=none -o libcudart.so