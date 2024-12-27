from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fwh',
    ext_modules=[
        CUDAExtension('fwh_cuda', [
            'fwh_cpp.cpp',
            'fwh_cu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })