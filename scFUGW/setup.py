from setuptools import setup, find_packages
import os
from pathlib import Path

#here = os.path.abspath(os.path.dirname(__file__))
#fugw_path = os.path.join(here, "fugw-main")

here = Path(os.path.abspath(os.path.dirname(__file__)))
fugw_uri = here.joinpath("fugw").absolute().as_uri()

setup(
    name="scFUGW",            
    version="0.1.3",
    packages=find_packages(),
    description="OT-based algorithm for single cell multimodal intergration",
#    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kai Peng",
    python_requires=">=3.7, <3.11",
    install_requires=[             
        "numpy>=1.18.0,<2.0",
        "pandas>=1.0.0",
        "anndata==0.10.5",
#        "fugw",
        "mudata",
        "scanpy",
        "muon",
        "scikit-learn",
        "scopen @ git+https://github.com/CostaLab/scopen@master#egg=scopen",
#        f"fugw @ file://{fugw_path}",
        f"fugw @ {fugw_uri}",
        
    ],
    # python_requires=">=3.6",
#    url="https://github.com/yourname/my_package",
    classifiers=[                  
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
