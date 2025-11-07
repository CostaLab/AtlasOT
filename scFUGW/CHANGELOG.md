# Changelog


## [0.1.3] - 2025-08-14

### Major Changes


给fugw的loss加上laplacian 正则化项


## [0.1.2] - 2025-07-11

### Major Changes


由于要给fugw的内部代码fit增加一个M计算矩阵参数，所以干脆将fugw包纳入我的文件下，这样直接修改后本地安装


## [0.1.1] - 2025-06-23

### Major Changes

- evaluate.py更新了四个RNA-Spatial的评估函数：PCC，SSIM，RMSE，JS
- 原RNA-ATAC代码scFUGW尽量保持不变，新的RNA-Spatial scFUGW使用新的函数scFUGW_RNA_Spatial


### Breaking Changes

### Bug Fixes



# Install

```
conda create -n scfugw python=3.10

conda activate scfugw

pip install .
```
