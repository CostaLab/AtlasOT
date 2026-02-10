from setuptools import setup, find_packages
import os
from pathlib import Path


here = Path(os.path.abspath(os.path.dirname(__file__)))
fugw_uri = here.joinpath("fugw").absolute().as_uri()

setup(
    name="scFUGW",            
    version="0.1.6",
    packages=find_packages(),
    description="OT-based algorithm for single cell multimodal intergration",
    long_description_content_type="text/markdown",
    author="Kai Peng",
    python_requires=">=3.7, <3.11",
    install_requires=[             
        "numpy>=1.18.0,<2.0",
        "pandas>=1.0.0",
        "anndata==0.11.4",
        "scanpy==1.11.5",
        "muon==0.1.7",
        "scikit-learn==1.7.2",
        "scopen @ git+https://github.com/CostaLab/scopen@master#egg=scopen",
        f"fugw @ {fugw_uri}",
        
    ],
    classifiers=[                  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
