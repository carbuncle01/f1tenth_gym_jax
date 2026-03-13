from setuptools import setup, find_packages

setup(
    name='f110_jax',
    version='0.2.0',
    description='JAX GPU-accelerated F1TENTH simulator — faithful reimplementation of f1tenth_gym',
    packages=find_packages(),
    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'scipy',
        'Pillow',
        'PyYAML',
        'matplotlib',
    ],
    dependency_links=[
        'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
    ],
    python_requires='>=3.8',
)