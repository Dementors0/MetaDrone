from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quadsim_cuda',
    ext_modules=[
        CUDAExtension('quadsim_cuda', [
            'quadsim.cpp',
            'quadsim_kernel.cu',
            'dynamics_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
