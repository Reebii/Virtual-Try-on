from setuptools import setup, find_packages

setup(
    name="virtual-tryon",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.3.0",
        "pillow>=10.0.0", 
        "python-dotenv>=1.0.0",
        "requests>=2.25.0",
        "pydantic>=1.10.0",
    ],
    python_requires=">=3.8",
)