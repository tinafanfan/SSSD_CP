from setuptools import setup
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'cauchy_mult', [
            'cauchy.cpp',
            'cauchy_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
                            # 'nvcc': ['-O2', '-lineinfo']
                            'nvcc': ['-O2', '-lineinfo', '--use_fast_math']
                            }
    )
    ext_modules.append(extension)
    print("hey")

setup(
    name='cauchy_mult',
    ext_modules=ext_modules,
    # cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
    cmdclass={'build_ext': BuildExtension})

# ============================================
# used in 140.109.74.45 Ubuntu 18.04

# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
# from torch.utils.cpp_extension import CUDA_HOME
# import torch
# import torch.cuda

# ext_modules = []
# if torch.cuda.is_available() and CUDA_HOME is not None:
#     extension = CUDAExtension(
#         'cauchy_mult', [
#             'cauchy.cpp',
#             'cauchy_cuda.cu',
#         ],
#         extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops'],
#                             'nvcc': ['-O2', '-lineinfo', '--use_fast_math']}
#     )
#     ext_modules.append(extension)
#     print("hey")

# setup(
#     name='cauchy_mult',
#     ext_modules=ext_modules,
#     install_requires=[
#         'torch',  # Add other dependencies if needed
#     ],
# )
# ============================================
