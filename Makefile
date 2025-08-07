libcudart.so:
	nvcc csrc/cudart.cc csrc/allocator.cc --compiler-options '-fPIC' -shared --cudart=none -lcuda -o libcudart.so

clean:
	rm -f libcudart.so

.PHONY: clean