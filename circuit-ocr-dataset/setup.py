"""Setup script for Circuit Schematic OCR."""

from setuptools import setup, find_packages

setup(
    name="circuit-ocr",
    version="1.0.0",
    description="Circuit Schematic OCR: PaddleOCR-VL based circuit understanding and netlist extraction",
    author="CircuitOCR Team",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "kiutils>=1.4.0",
        "sexpdata>=1.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "cairosvg>=2.7.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "paddle": ["paddlepaddle-gpu>=2.6.0", "paddleocr>=2.7.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
