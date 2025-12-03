from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="beyond_the_board",
    version="0.1.0",
    author="Beyond The Board Team",
    description="Chess position evaluation using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CousinCombey/beyond_the_board",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Le Wagon Students",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "python-chess>=1.999",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "jupyter": [
            "ipywidgets",
            "cairosvg",
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
    },
)
