"""
PyTorch CUDA Memory Allocator

Custom CUDA memory allocator that can be used as a drop-in replacement for CUDART.
"""

import os
import ctypes

__version__ = "0.1.0"

# Get the path to the shared library
_lib_path = os.path.join(os.path.dirname(__file__), 'libcudart.so')

def get_library_path():
    """Get the path to the libcudart.so shared library"""
    if os.path.exists(_lib_path):
        return _lib_path
    else:
        raise FileNotFoundError(f"libcudart.so not found at {_lib_path}")

def load_library():
    """Load the libcudart.so shared library"""
    lib_path = get_library_path()
    return ctypes.CDLL(lib_path)

# Instructions for usage
def usage_info():
    """Print usage information"""
    lib_path = get_library_path()
    print(f"To use the custom CUDA allocator, set the LD_PRELOAD environment variable:")
    print(f"export LD_PRELOAD={lib_path}")
    print(f"Or run your Python script with:")
    print(f"LD_PRELOAD={lib_path} python your_script.py")
