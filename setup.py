from setuptools import setup, find_packages

setup(
    name='neurogenesis',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm',
        'IPython',
        'torch',
        'torchvision',
    ],
)
