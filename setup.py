from setuptools import setup, find_packages

setup(
    name='hf_interp',
    version='0.1.0',
    url='https://github.com/slavachalnev/hf-interp',
    author='Sviatoslav Chalnev',
    description='Hooked Transformer interpretability based on HuggingFace Transformers',
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "einops",
        "fancy_einsum",
        "tqdm",
        "jaxtyping",
        "pytest"
    ],
)