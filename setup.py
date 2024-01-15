from setuptools import setup, find_packages

setup(
    name="gpu-multi-tenancy",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "psutil>=5.8.0",
        "pynvml>=11.0.0",
    ],
    author="Erfan Darzi",
    description="GPU multi-tenancy controller with dynamic MIG and PCIe-aware placement",
    python_requires=">=3.8",
)