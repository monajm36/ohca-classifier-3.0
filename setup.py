from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ohca-classifier-v3",
    version="3.0.0",
    author="Mona Moukaddem",
    author_email="mona.moukaddem@example.com",  # Update with your actual email
    description="BERT-based classifier for detecting Out-of-Hospital Cardiac Arrest (OHCA) cases in medical discharge notes with enhanced v3.0 methodology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/monajm36/ohca-classifier-3.0",
    project_urls={
        "Bug Reports": "https://github.com/monajm36/ohca-classifier-3.0/issues",
        "Source": "https://github.com/monajm36/ohca-classifier-3.0",
        "Documentation": "https://github.com/monajm36/ohca-classifier-3.0/blob/main/README.md",
        "Model": "https://huggingface.co/monajm36/ohca-classifier-v3",
    },
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Text Processing :: Linguistic",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    keywords=[
        "medical-nlp", "cardiac-arrest", "ohca", "bert", "clinical-decision-support",
        "medical-ai", "healthcare", "nlp", "transformers", "mimic", "pubmedbert"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "all": [
            "pytest>=6.0",
            "black>=22.0", 
            "flake8>=4.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ohca-train=scripts.train_from_labeled_data:main",
            "ohca-predict=scripts.predict_ohca:main", 
            "ohca-prepare=scripts.prepare_data:main",
        ],
    },
    package_data={
        "src": ["*.json", "*.txt"],
    },
    include_package_data=True,
    zip_safe=False,
)
