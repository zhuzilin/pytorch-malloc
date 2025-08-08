#!/usr/bin/env python3

import os
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install


class CustomBuildExt(build_ext):
    """Custom build extension to run make command"""
    
    def run(self):
        # Get the source directory (where this setup.py is located)
        source_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Run make to build the shared library
        try:
            print("Building libcudart.so...")
            # Clean first
            subprocess.check_call(['make', 'clean'], cwd=source_dir)
            # Build the shared library
            subprocess.check_call(['make'], cwd=source_dir)
            print("Successfully built libcudart.so")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build shared library: {e}")
        except FileNotFoundError:
            raise RuntimeError("Make command not found. Please ensure 'make' is installed.")
        
        # Continue with the normal build process
        super().run()


class CustomInstall(install):
    """Custom install command to ensure libcudart.so is in the right place"""
    
    def run(self):
        # Run the normal install process first
        super().run()
        
        # Get source directory
        source_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Find where the package was installed
        package_dir = os.path.join(self.install_lib, 'pytorch_malloc')
        
        # Copy the shared library to the package directory
        src_lib = os.path.join(source_dir, 'libcudart.so')
        if os.path.exists(src_lib):
            if not os.path.exists(package_dir):
                os.makedirs(package_dir)
            dst_lib = os.path.join(package_dir, 'libcudart.so')
            shutil.copy2(src_lib, dst_lib)
            print(f"Copied {src_lib} to {dst_lib}")
        else:
            print(f"Warning: {src_lib} not found. Make sure the build was successful.")


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name="pytorch-malloc",
    version="0.1.0",
    author="zhuzilin",
    description="Custom CUDA memory allocator for PyTorch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/zhuzilin/pytorch-malloc",
    packages=['pytorch_malloc'],
    package_dir={'pytorch_malloc': '.'},
    package_data={
        'pytorch_malloc': ['libcudart.so', '*.py'],
    },
    include_package_data=True,
    cmdclass={
        'build_ext': CustomBuildExt,
        'install': CustomInstall,
    },
    ext_modules=[Extension('dummy', sources=[])],  # Dummy extension to trigger build_ext
    python_requires='>=3.6',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,  # Important: don't zip the package since we need the .so file
)
