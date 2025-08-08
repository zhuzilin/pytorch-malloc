build:

```bash
make clean && make
```

test:

```bash
LD_PRELOAD=./libcudart.so python tests/test_offload.py
```