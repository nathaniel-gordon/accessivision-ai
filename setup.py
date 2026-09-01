from setuptools import setup, find_packages

setup(
    name="accessivision-ai",
    version="1.0.0",
    description="Fairness-Aware Multimodal Generative Models: LoRA Diffusion & QLoRA LLMs for Assistive Vision and On-Device CoreML Inference",
    author="Nathaniel Gordon",
    author_email="nathanielgordon346@gmail.com",
    url="https://github.com/nathaniel-gordon/accessivision-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "tensorflow>=2.15.0",
        "transformers>=4.40.0",
        "diffusers>=0.27.0",
        "peft>=0.10.0",
        "coremltools>=7.2",
        "Pillow>=10.2.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
)
