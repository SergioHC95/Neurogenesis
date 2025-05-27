from setuptools import setup, find_packages

setup(
    name='grow_when_needed',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch',
        'numpy',
        'matplotlib',
        'tqdm'
    ],
)
