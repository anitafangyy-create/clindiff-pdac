from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clindiff-pdac",
    version="0.1.0",
    author="ClinDiff-PDAC Team",
    author_email="contact@clindiff-pdac.org",
    description="Clinical Knowledge-Guided Diffusion Model for Pancreatic Cancer EMR Imputation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/clindiff-pdac",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "llm": [
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "peft>=0.4.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "clindiff-train=clindiff_pdac.scripts.train:main",
            "clindiff-impute=clindiff_pdac.scripts.impute:main",
            "clindiff-evaluate=clindiff_pdac.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "clindiff_pdac": ["data/*", "configs/*"],
    },
)
