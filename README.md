# SoftCTC-GPU
GPU implementation of SoftCTC loss. See [SoftCTC repository](https://github.com/DCGM/SoftCTC) for more information. 

## Required libs
- ```cuda 11.X``` (tested on 11.8, needed only with Cuda flag enabled)
- ```cudnn``` (tested on 8.8, needed only for cuda enabled pytorch)
- ```pytorch with cuda 11.X support``` (cuda enabled pytorch is needed only for copyless passing of cuda buffers)
- ```OpenCL >=1.1``` (needed only with OpenCL flag enabled)
- ```CMake >=3.24```

## Supported platforms
- Nvidia with SM >=6.1 (Polaris, Turing, Ampere, Ada Lovelace, ... older platforms is not supported because of lack of ```atomicAdd``` instructions on double datatype) using inline assembler
- AMD/Intel platform is currently not supported unless ```__opencl_c_ext_fp32_local_atomic_add``` and ```__opencl_c_ext_fp64_local_atomic_add``` is supported

## Compilation instructions
```shell
mkdir build
cd build
cmake ../ <options>
``` 

### CMake options

- Linux CMake default paths (for Python 3.10 and installed PyTorch by pip using sudo):
```shell
-DTorch_DIR="/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/" 
-DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc"
```

- Windows CMake default paths (for Python 3.10 and Cudnn 8.8):
```shell
-DTorch_DIR="C:\Python310\Lib\site-packages\torch\share\cmake\Torch" 
-Dpybind11_DIR="C:\Python310\Lib\site-packages\pybind11\share\cmake\pybind11" 
-DCUDNN_ROOT="C:\Program Files\NVIDIA\CUDNN\v8.8"
```
- Additional options for CMake:
```shell
-DENABLE_OPENCL=ON          enable compilation of OpenCL implementation (add CTCOpenCLFloat/CTCOpenCLDouble classes)
-DENABLE_CUDA=OFF           disable compilation of Cuda implementation
-DENABLE_TORCH=OFF          disable support for torch buffers
-DENABLE_COMPUTE_CACHE=OFF  disable OpenCL Cache
-DENABLE_PYBIND=OFF         disable PyBind (Compile as executable)
-DENABLE_PYBIND=OFF         enable printing of Cuda/OpenCL kernel and copy times (On Cuda available only with sync_native argument of CTCCudaFloat/CTCCudaDouble classes set to true)
 ```
