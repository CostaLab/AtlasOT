#!/usr/bin/env python3
import scanpy as sc
import sys
import os

def check_sample(sample_name):
    """检查单个样本的数据shape"""
    rna_path = f"../../../DataCenter/HTAPP/cleaned_dataset/{sample_name}_rna.h5ad"
    spatial_path = f"../../../DataCenter/HTAPP/cleaned_dataset/{sample_name}_spatial.h5ad"
    
    if not os.path.exists(rna_path) or not os.path.exists(spatial_path):
        print(f"样本 {sample_name} 文件不存在")
        return
    
    rna = sc.read_h5ad(rna_path)
    spatial = sc.read_h5ad(spatial_path)
    
    print(f"{sample_name}:")
    print(f"  RNA shape: {rna.X.shape}")
    print(f"  Spatial shape: {spatial.X.shape}")
    print()

def main():
    # 获取所有样本
    data_dir = "../../../DataCenter/HTAPP/cleaned_dataset"
    rna_files = [f for f in os.listdir(data_dir) if f.endswith('_rna.h5ad')]
    samples = [f.replace('_rna.h5ad', '') for f in rna_files]
    
    print(f"找到 {len(samples)} 个样本")
    print("=" * 50)
    
    for sample in sorted(samples):
        check_sample(sample)

if __name__ == "__main__":
    main()
