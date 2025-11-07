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


def load_smfish_data():
    """
    加载smFISH数据集
    """
    # 数据路径
    data_path = "../../../DataCenter/smFISH/smFISH.h5mu.gz"
    
    print("正在加载smFISH数据集...")
    
    # 使用muon读取MuData
    m = mu.read_h5mu(data_path)
    
    print(f"MuData对象信息: {m}")
    print(f"fish modality形状: {m['fish'].shape}")
    print(f"rna modality形状: {m['rna'].shape}")
    
    # 提取fish和rna数据
    fish = m['fish'].copy()
    rna = m['rna'].copy()
    
    return fish, rna


def prepare_spatial_coordinates(spatial_data):
    """
    将spatial数据的坐标信息从obs保存到obsm中
    """
    print("正在处理spatial坐标信息...")
    
    # 检查是否有坐标信息
    if 'x_coord' in spatial_data.obs.columns and 'y_coord' in spatial_data.obs.columns:
        # 提取坐标
        x_coords = spatial_data.obs['x_coord'].values
        y_coords = spatial_data.obs['y_coord'].values
        
        # 保存到obsm中
        spatial_data.obsm['spatial'] = np.column_stack([x_coords, y_coords])
        print(f"已将坐标信息保存到obsm['spatial']，形状: {spatial_data.obsm['spatial'].shape}")
    else:
        print("警告：未找到x_coord和y_coord列")
    
    return spatial_data


def preprocess_data(rna, spatial):
    """
    预处理数据
    """
    print("正在预处理数据...")

    def find_threshold(values, threshold = 1e-3):
        for i, v in enumerate(values):
            if v < threshold:
                print(i, v)
                return i

    # Preprocessing RNA
    def preprocess_rna(rna, save_counts = True, filter=False):
        rna.var.index = rna.var.index.str.split(':', expand = True)
        rna.var['features'] = rna.var.index
        if filter == True:
            sc.pp.filter_cells(rna, min_genes = 1)
            sc.pp.filter_genes(rna, min_counts = 10)
        non_mito_genes_list = [name for name in rna.var_names if not name.startswith('MT-')]
        rna = rna[:, non_mito_genes_list]

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

    # 预处理RNA数据
    sc.pp.filter_cells(rna, min_genes=200)
    rna = preprocess_rna(rna, save_counts=True)
    rna = reduction_rna(rna, sample_id=None, random_seed=3407)
    
    # 预处理spatial数据
    spatial = preprocess_spatial(spatial, save_counts=True)
    spatial = reduction_spatial(spatial, sample_id=None, random_seed=3407)
    
    print("数据预处理完成")
    return rna, spatial


def run_single_fold(fold_idx, test_genes, train_genes, rna, fish, method):
    """
    运行单个fold的测试
    """
    print(f"\n{'='*20} Fold {fold_idx+1} - {method.upper()} {'='*20}")
    print(f"测试基因: {test_genes}")
    print(f"训练基因数量: {len(train_genes)}")
    
    # 创建Benchmark实例
    benchmark = Benchmark(
        rna=rna,
        sp=fish,
        test_genes=test_genes,
        train_genes=train_genes,
        device='cpu'
    )
    
    # 获取真实的空间基因表达
    fish_test = fish[:, test_genes].copy()
    real_result = fish_test.to_df()
    
    # 运行指定方法
    print(f"\n开始运行 {method} 方法 (Fold {fold_idx+1})...")
    imputed_result = benchmark.run_single_method(method)
    
    if imputed_result is not None:
        print(f"\n正在评估 {method}...")
        try:
            result = evaluate_method(f'{method}-Fold{fold_idx+1}', imputed_result, real_result)
            return result
        except Exception as e:
            print(f"评估 {method} 时出错: {e}")
            return None
    else:
        print(f"{method} 推断失败或跳过")
        return None


def save_results_to_json(method, all_fold_results, avg_results, num_folds, valid_folds_count, results_dir="results"):
    """
    保存结果到JSON文件
    """
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 准备保存的数据结构
    results_data = {
        "method": method.upper(),
        "timestamp": datetime.now().isoformat(),
        "total_folds": num_folds,
        "successful_folds": valid_folds_count,
        "fold_results": [],
        "average_results": avg_results,
        "summary": {
            "PCC_mean": float(np.float64(avg_results.get('PCC', 0))),
            "JS_mean": float(np.float64(avg_results.get('JS', 0))),
            "RMSE_mean": float(np.float64(avg_results.get('RMSE', 0))),
            "SSIM_mean": float(np.float64(avg_results.get('SSIM', 0)))
        }
    }
    
    # 添加每个fold的结果
    for fold_idx, result in enumerate(all_fold_results):
        fold_data = {
            "fold": fold_idx + 1,
            "success": result is not None
        }
        
        if result is not None:
            fold_data.update({
                "PCC": float(np.float64(result['PCC'])),
                "JS": float(np.float64(result['JS'])),
                "RMSE": float(np.float64(result['RMSE'])),
                "SSIM": float(np.float64(result['SSIM']))
            })
        else:
            fold_data.update({
                "PCC": None,
                "JS": None,
                "RMSE": None,
                "SSIM": None
            })
        
        results_data["fold_results"].append(fold_data)
    
    # 保存到JSON文件
    filename = f"smFISH-{method}-result.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {filepath}")
    return filepath


def main():
    """
    主函数：执行smFISH数据集的K-fold交叉验证
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='smFISH数据集K-fold交叉验证基准测试')
    parser.add_argument('--method', type=str, required=True, 
                       choices=['spage', 'tangram', 'gimvi', 'novosparc', 'spaotsc', 'stplus', 'scfugw'],
                       help='要运行的算法名称')
    parser.add_argument('--folds', type=int, default=None,
                       help='要运行的fold数量，默认运行所有fold')
    parser.add_argument('--results_dir', type=str, default='./P001-smFISH',
                       help='结果保存目录，默认为./P001-smFISH')
    
    args = parser.parse_args()
    method = args.method.lower()
    max_folds = args.folds
    results_dir = args.results_dir
    
    print(f"开始smFISH数据集K-fold交叉验证 - {method.upper()}算法基准测试")
    print("="*80)
    
    # 加载数据
    fish, rna = load_smfish_data()
    
    # 处理spatial坐标信息
    fish = prepare_spatial_coordinates(fish)
    
    # 预处理数据
    rna, fish = preprocess_data(rna, fish)
    
    # 加载K-fold基因选择
    test_genes_all = np.load("../../../DataCenter/smFISH/test_genes/test_genes.npy", allow_pickle=True)
    train_genes_all = np.load("../../../DataCenter/smFISH/test_genes/train_genes.npy", allow_pickle=True)
    
    print(f"加载了{len(test_genes_all)}个fold的基因选择")
    
    # 确定要运行的fold数量
    if max_folds is not None:
        num_folds = min(max_folds, len(test_genes_all))
        print(f"将运行前{num_folds}个fold")
    else:
        num_folds = len(test_genes_all)
        print(f"将运行所有{num_folds}个fold")
    
    # 存储所有fold的结果
    all_fold_results = []
    
    # 运行每个fold
    for fold_idx in range(num_folds):
        test_genes = test_genes_all[fold_idx]
        train_genes = train_genes_all[fold_idx]
        
        fold_result = run_single_fold(fold_idx, test_genes, train_genes, rna, fish, method)
        all_fold_results.append(fold_result)
    
    # 汇总所有fold的结果
    print("\n" + "="*80)
    print(f"{method.upper()}算法K-fold交叉验证结果汇总")
    print("="*80)
    
    # 过滤掉None结果
    valid_results = [result for result in all_fold_results if result is not None]
    
    if len(valid_results) == 0:
        print(f"警告：{method}算法在所有fold中都失败了")
        return
    
    # 打印每个fold的详细结果
    print(f"\n{method.upper()}算法各fold结果:")
    print("-" * 60)
    print(f"{'Fold':<6} {'PCC':<8} {'JS':<8} {'RMSE':<8} {'SSIM':<8}")
    print("-" * 60)
    
    for fold_idx, result in enumerate(all_fold_results):
        if result is not None:
            print(f"{fold_idx+1:<6} {result['PCC']:<8.4f} {result['JS']:<8.4f} {result['RMSE']:<8.4f} {result['SSIM']:<8.4f}")
        else:
            print(f"{fold_idx+1:<6} {'失败':<8} {'失败':<8} {'失败':<8} {'失败':<8}")
    
    # 计算并打印平均结果
    print(f"\n{'='*20} {method.upper()}平均结果 {'='*20}")
    print(f"{'PCC':<8} {'JS':<8} {'RMSE':<8} {'SSIM':<8} {'成功fold数':<8}")
    print("-" * 50)
    
    # 计算平均结果
    avg_results = {}
    for metric in ['PCC', 'JS', 'RMSE', 'SSIM']:
        values = [result[metric] for result in valid_results]
        avg_results[metric] = float(np.mean(values))
    
    print(f"{avg_results['PCC']:<8.4f} {avg_results['JS']:<8.4f} {avg_results['RMSE']:<8.4f} {avg_results['SSIM']:<8.4f} {len(valid_results):<8}")
    
    # 保存结果到JSON文件
    print(f"\n{'='*20} 保存结果 {'='*20}")
    save_results_to_json(method, all_fold_results, avg_results, num_folds, len(valid_results), results_dir)
    
    print(f"\n{method.upper()}算法K-fold交叉验证完成!")
    print(f"成功完成 {len(valid_results)}/{num_folds} 个fold")


if __name__ == "__main__":
    main()
