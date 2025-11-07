import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as pp
from scipy.sparse import issparse
import argparse
import sys
import os
import json
from datetime import datetime

# 导入通用benchmark模块
from benchmark import Benchmark, evaluate_method


def load_mi_data(sample_name):
    """
    加载MI数据集
    """
    data_path = f"../../../DataCenter/MI/multimodal_data/{sample_name}.h5mu"
    m = mu.read_h5mu(data_path)
    rna = m['gene_expression'].copy()
    sp = m['spatial'].copy()
    return rna, sp


def prepare_spatial_coordinates(spatial_data):
    """
    处理MI数据集的spatial数据
    检查并转换坐标信息格式
    """
    # 检查是否已经有spatial坐标信息
    if 'spatial' in spatial_data.obsm:
        return spatial_data
    
    # 如果存在X_spatial，转换为spatial
    if 'X_spatial' in spatial_data.obsm:
        spatial_data.obsm['spatial'] = spatial_data.obsm['X_spatial']
        return spatial_data
    
    return spatial_data


def preprocess_data(rna, spatial):
    """
    预处理数据
    """
    def find_threshold(values, threshold = 1e-3):
        for i, v in enumerate(values):
            if v < threshold:
                return i

    # Preprocessing RNA
    def preprocess_rna(rna, save_counts = True, filter=False):
        rna.var.index = rna.var.index.str.split(':', expand = True)
        rna.var['features'] = rna.var.index
        if filter == True:
            sc.pp.filter_cells(rna, min_genes = 1)
            # sc.pp.filter_genes(rna, min_counts = 10)
        # non_mito_genes_list = [name for name in rna.var_names if not name.startswith('MT-')]
        # rna = rna[:, non_mito_genes_list]

        if save_counts == True:
            rna.layers["counts"] = rna.X

        sc.pp.normalize_total(rna, inplace=True)
        sc.pp.log1p(rna)
        return rna

    def reduction_rna(rna, sample_id = None, random_seed = 3407):
        pca_comps = min(rna.X.shape[0], rna.X.shape[1], 300)
        pca = PCA(n_components = pca_comps, random_state = random_seed)
        X = rna.X.toarray() if issparse(rna.X) else np.asarray(rna.X)
        pca = pca.fit(X)
        rna_pca_explained_var = pca.explained_variance_ratio_
        index = find_threshold(rna_pca_explained_var)
        rna.obsm["RNA_pca_l2_norm"] = pca.transform(X)[:, :index]
        rna.obsm["RNA_pca_l2_norm"] = pp(
            rna.obsm["RNA_pca_l2_norm"], norm="l2"
        )
        if sample_id is not None:
            try:
                import scanpy.external as sce
                sce.pp.harmony_integrate(rna, sample_id, "RNA_pca_l2_norm", adjusted_basis = 'RNA_pca_l2_norm')
            except ImportError:
                print("警告：scanpy.external不可用，跳过harmony整合")
        return rna

    def preprocess_spatial(spatial, save_counts = True):
        spatial.var.index = spatial.var.index.str.split(':', expand = True)
        spatial.var['features'] = spatial.var.index

        if save_counts == True:
            spatial.layers["counts"] = spatial.X

        sc.pp.normalize_total(spatial, inplace=True)
        sc.pp.log1p(spatial)
        return spatial

    def reduction_spatial(spatial, sample_id = None, random_seed = 3407):
        pca_comps = min(spatial.X.shape[0], spatial.X.shape[1], 300)
        pca = PCA(n_components = pca_comps, random_state = random_seed)
        X = spatial.X.toarray() if issparse(spatial.X) else np.asarray(spatial.X)
        pca = pca.fit(X)
        spatial_pca_explained_var = pca.explained_variance_ratio_
        index = find_threshold(spatial_pca_explained_var)
        spatial.obsm["spatial_pca_l2_norm"] = pca.transform(X)[:, :index]
        spatial.obsm["spatial_pca_l2_norm"] = pp(
            spatial.obsm["spatial_pca_l2_norm"], norm="l2"
        )
        if sample_id is not None:
            try:
                import scanpy.external as sce
                sce.pp.harmony_integrate(spatial, sample_id, "spatial_pca_l2_norm", adjusted_basis = 'spatial_pca_l2_norm')
            except ImportError:
                print("警告：scanpy.external不可用，跳过harmony整合")
        return spatial

    # 预处理数据
    sc.pp.filter_cells(rna, min_genes=200)
    rna = preprocess_rna(rna, save_counts=True)
    rna = reduction_rna(rna, sample_id=None, random_seed=3407)
    
    spatial = preprocess_spatial(spatial, save_counts=True)
    spatial = reduction_spatial(spatial, sample_id=None, random_seed=3407)
    
    return rna, spatial


def run_single_fold(test_genes, train_genes, rna, sp, method):
    """
    运行单个fold的测试
    """
    benchmark = Benchmark(
        rna=rna,
        sp=sp,
        test_genes=test_genes,
        train_genes=train_genes,
        device=None,
    )
    
    sp_test = sp[:, test_genes].copy()
    real_result = sp_test.to_df()
    
    imputed_result = benchmark.run_single_method(method)
    
    if imputed_result is not None:
        try:
            result = evaluate_method(f'{method}', imputed_result, real_result)
            return result, imputed_result, real_result
        except Exception as e:
            print(f"评估 {method} 时出错: {e}")
            return None, None, None
    else:
        return None, None, None


def save_final_results(method, all_sample_results, dataset_avg, results_dir="./P002-MI"):
    """
    保存最终结果到JSON文件
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备保存的数据结构
    results_data = {
        "method": method.upper(),
        "timestamp": datetime.now().isoformat(),
        "dataset_average": dataset_avg,
        "total_samples": len(all_sample_results),
        "total_folds_per_sample": 10,
        "sample_results": {}
    }
    
    # 添加每个样本的结果
    for sample_name, sample_data in all_sample_results.items():
        results_data["sample_results"][sample_name] = {
            "sample_average": sample_data['avg_result'],
            "fold_results": sample_data['fold_results']
        }
    
    # 保存到JSON文件
    filename = f"MI-{method}-final-results-scfugw-common-test.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    return filepath


def get_available_samples():
    """
    自动获取所有可用的样本文件
    """
    data_dir = "../../../DataCenter/MI/multimodal_data"
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在: {data_dir}")
        exit(1)
    
    # 获取所有.h5mu文件
    h5mu_files = [f for f in os.listdir(data_dir) if f.endswith('.h5mu')]
    # 移除.h5mu扩展名获取样本名
    samples = [f.replace('.h5mu', '') for f in h5mu_files]
    return sorted(samples)

def main():
    """
    主函数：执行MI数据集的所有样本测试
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MI数据集所有样本测试')
    parser.add_argument('--method', type=str, required=True, 
                       choices=['spage', 'tangram', 'gimvi', 'novosparc', 'spaotsc', 'stplus', 'scfugw'],
                       help='要运行的算法名称')
    parser.add_argument('--results_dir', type=str, default='./P002-MI',
                       help='结果保存目录，默认为./P002-MI')
    
    args = parser.parse_args()
    method = args.method.lower()
    results_dir = args.results_dir
    
    # 自动获取所有样本
    samples = get_available_samples()
    if not samples:
        print("错误：未找到任何样本文件")
        return
    
    print(f"找到 {len(samples)} 个样本: {samples}")
    
    # 存储所有样本的结果
    all_sample_results = {}
    
    # 循环处理所有样本
    for i, sample_name in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"处理样本 {i+1}/{len(samples)}: {sample_name}")
        print(f"{'='*60}")

        
        # 加载数据
        rna, sp = load_mi_data(sample_name)
        sp = prepare_spatial_coordinates(sp)
        rna, sp = preprocess_data(rna, sp)
        
        # 加载测试基因 (10-fold交叉验证)
        test_genes_all_folds = np.load(f"../../../DataCenter/MI/test_squidgenes/{sample_name[16:]}_test.npy", allow_pickle=True)
        
        # 获取所有基因
        all_genes = list(sp.var_names)
        
        # 运行10-fold交叉验证
        all_fold_results = []
        for fold_idx in range(test_genes_all_folds.shape[0]):
            print(f"\n处理Fold {fold_idx+1}/10...")
            
            # 获取当前fold的测试基因
            test_genes = test_genes_all_folds[fold_idx]
            
            # 获取训练基因（排除当前fold的测试基因，且必须在RNA基因中）
            train_genes = [gene for gene in all_genes if gene not in test_genes]







            # 这里记得删掉----------------------------------------------------------------------------------------
            # 这里是为了测试在train是common和非common下的scfugw的表现
            # train_genes = [gene for gene in all_genes if gene not in test_genes and gene in rna.var_names]
            # 这里记得删掉----------------------------------------------------------------------------------------







            # 运行测试
            result, imputed_result, real_result = run_single_fold(test_genes, train_genes, rna, sp, method)
            
            if result is not None:
                all_fold_results.append(result)
                print(f"Fold {fold_idx+1} 结果: PCC={result['PCC']:.4f}, JS={result['JS']:.4f}, RMSE={result['RMSE']:.4f}, SSIM={result['SSIM']:.4f}")
            else:
                print(f"Fold {fold_idx+1} 测试失败!")
                exit(1)
        
        # 计算平均结果
        if all_fold_results:
            avg_result = {}
            for metric in ['PCC', 'JS', 'RMSE', 'SSIM']:
                values = [result[metric] for result in all_fold_results]
                avg_result[metric] = float(np.mean(values))
            
            print(f"\n{method.upper()}算法平均结果 ({sample_name}):")
            print(f"PCC: {avg_result['PCC']:.4f}, JS: {avg_result['JS']:.4f}, RMSE: {avg_result['RMSE']:.4f}, SSIM: {avg_result['SSIM']:.4f}")
            
            # 存储样本结果
            all_sample_results[sample_name] = {
                'avg_result': avg_result,
                'fold_results': all_fold_results
            }
        else:
            print(f"样本 {sample_name} 所有fold都失败!")
            exit(1)
        
        print(f"样本 {sample_name} 测试完成!")
    
    # 计算并保存最终结果
    print(f"\n{'='*60}")
    print(f"计算数据集总平均结果...")
    print(f"{'='*60}")
    
    # 计算数据集总平均
    dataset_avg = {}
    for metric in ['PCC', 'JS', 'RMSE', 'SSIM']:
        values = [sample_result['avg_result'][metric] for sample_result in all_sample_results.values()]
        dataset_avg[metric] = float(np.mean(values))
    
    print(f"\n{method.upper()}算法数据集总平均结果:")
    print(f"PCC: {dataset_avg['PCC']:.4f}, JS: {dataset_avg['JS']:.4f}, RMSE: {dataset_avg['RMSE']:.4f}, SSIM: {dataset_avg['SSIM']:.4f}")
    
    # 保存最终结果
    final_result_path = save_final_results(method, all_sample_results, dataset_avg, results_dir)
    print(f"\n最终结果已保存到: {final_result_path}")
    
    print(f"\n{'='*60}")
    print(f"所有样本处理完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
