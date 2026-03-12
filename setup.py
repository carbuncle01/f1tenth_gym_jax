from setuptools import setup, find_packages

setup(
    name='f110_jax',
    version='0.2.0',
    description='JAX GPU-accelerated F1TENTH simulator — faithful reimplementation of f1tenth_gym',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'numpy',
        'scipy',
        'Pillow',
        'PyYAML',
        'matplotlib',
    ],
    python_requires='>=3.8',
)
