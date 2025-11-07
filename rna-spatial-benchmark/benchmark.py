import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
import muon as mu
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as pp
from scipy import stats as st


class Benchmark:
    """
    统一的基准评测类（仅需测试基因）。
    - 输入：AnnData rna, AnnData sp, test_genes, train_genes；可选 outdir/device
    - 方法：spage, tangram, gimvi, novosparc, spaotsc, stplus
    - 返回：DataFrame (spots x genes) 或 None（依赖缺失/前置条件不满足）
    """

    def __init__(self, rna, sp, test_genes: List[str], train_genes: List[str] = None, outdir: Optional[str] = None, device: Optional[str] = None):
        self.rna = rna
        self.sp = sp
        self.test_genes = list(test_genes)
        self.train_genes = list(train_genes) if train_genes is not None else []
        self.outdir = outdir
        # 自动检测设备：优先GPU，如果不可用则使用CPU
        self.device = self._detect_device(device)

    def _detect_device(self, device: Optional[str] = None) -> str:
        """自动检测可用的计算设备，优先使用GPU"""
        if device is not None:
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"使用GPU进行计算")
                return 'cuda:0'
            else:
                print("使用CPU进行计算")
                return 'cpu'
        except ImportError:
            return 'cpu'
        except Exception:
            return 'cpu'

    def spage(self) -> Optional[pd.DataFrame]:
        try:
            from SpaGE.main import SpaGE
        except Exception as e:
            print(f"SpaGE导入失败: {e}")
            return None
        
        # 准备数据
        rna_df = pd.DataFrame(self.rna.X.toarray() if hasattr(self.rna.X, 'toarray') else np.asarray(self.rna.X),
                              index=self.rna.obs_names, columns=self.rna.var_names).T
        sp_df = pd.DataFrame(self.sp.X.toarray() if hasattr(self.sp.X, 'toarray') else np.asarray(self.sp.X),
                             index=self.sp.obs_names, columns=self.sp.var_names)
        
        # 过滤零方差基因
        rna_df = rna_df.loc[(rna_df.sum(axis=1) != 0) & (rna_df.var(axis=1) != 0)]
        
        # 检查基因可用性
        predict = [g for g in self.test_genes if g in rna_df.index and g in sp_df.columns]
        feature = [g for g in self.train_genes if g in rna_df.index and g in sp_df.columns]
        if len(predict) == 0 or len(feature) == 0:
            return None
        
        # 准备数据子集
        sp_subset = sp_df[feature]
        rna_subset = rna_df.T
        
        # 计算PCA组件数
        n_pv = min(sp_subset.shape[1], rna_subset.shape[0], sp_subset.shape[0], 100) - 1
        n_pv = max(1, min(n_pv, len(feature) // 2))

        if n_pv < 100:
            similarity_threshold = 0.1
            print('**********************************************')
            print('SpaGE使用相似度阈值: 0.1')
            print('**********************************************')
        else:
            similarity_threshold = 0.3
        
        
        res = SpaGE(sp_subset, rna_subset, n_pv=n_pv, genes_to_predict=predict, similarity_threshold=similarity_threshold)
        
        # 修复索引问题：SpaGE返回的DataFrame使用默认整数索引，需要设置为原始spatial数据的索引
        result = res[predict]
        if result is not None:
            # 使用原始spatial数据的索引
            result.index = sp_df.index
            print(f"SpaGE修复后结果形状: {result.shape}")
            print(f"SpaGE修复后结果索引前5个: {list(result.index[:5])}")
        
        return result

    def tangram(self) -> Optional[pd.DataFrame]:
        try:
            import tangram as tg
            import torch
        except Exception as e:
            print(f"Tangram导入失败: {e}")
            return None
        
        # 复制数据
        RNA_data_adata = self.rna.copy()
        Spatial_data_adata = self.sp[:, self.train_genes].copy()
        
        # Tangram预处理
        tg.pp_adatas(RNA_data_adata, Spatial_data_adata, genes=self.train_genes)
        
        # 执行映射
        dev = torch.device(self.device) if self.device else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        ad_map = tg.map_cells_to_space(RNA_data_adata, Spatial_data_adata, device=dev)
        ad_ge = tg.project_genes(ad_map, RNA_data_adata)
        
        # 将投影结果中的基因名从小写改回大写
        ad_ge.var_names = [name.upper() for name in ad_ge.var_names]
        
        # 提取测试基因结果
        test = [g for g in self.test_genes if g in ad_ge.var_names]
        if len(test) == 0:
            return None
        
        return pd.DataFrame(ad_ge[:, test].X, index=ad_ge.obs_names, columns=test)

    def gimvi(self, max_epochs: int = 200) -> Optional[pd.DataFrame]:
        try:
            import scvi
            # from scvi.model import GIMVI
            from scvi.external import GIMVI
            import scanpy as sc
        except Exception as e:
            print(f"GIMVI导入失败: {e}")
            return None
        
        # 准备基因列表
        genes_all = list(dict.fromkeys(self.train_genes + self.test_genes))
        genes_all = [g for g in genes_all if g in self.rna.var_names]
        train_present = [g for g in self.train_genes if g in self.rna.var_names]
        
        if len(genes_all) == 0 or len(train_present) == 0:
            return None
        
        # 准备数据
        spatial_data_partial = self.sp[:, train_present].copy()
        seq_data = self.rna[:, genes_all].copy()
        
        # 设置数据
        GIMVI.setup_anndata(spatial_data_partial)
        GIMVI.setup_anndata(seq_data)
        
        # 训练模型
        model = GIMVI(seq_data, spatial_data_partial)
        
        # 设置设备
        if self.device and 'cuda' in self.device:
            model.to_device(self.device)
            print(f"GIMVI使用GPU设备: {self.device}")
        else:
            print("GIMVI使用CPU设备")
        
        model.train(max_epochs)
        
        # 获取推断结果
        _, imputation = model.get_imputed_values(normalized=False)
        test_present = [g for g in self.test_genes if g in seq_data.var_names]
        if len(test_present) == 0:
            return None
        
        test_idx = [seq_data.var_names.tolist().index(g) for g in test_present]
        imputed = imputation[:, test_idx]
        
        return pd.DataFrame(imputed, index=spatial_data_partial.obs_names, columns=test_present)

    def novosparc(self) -> Optional[pd.DataFrame]:
        try:
            import novosparc as nc
            from scipy.spatial.distance import cdist
        except Exception as e:
            print(f"NovoSpaRC导入失败: {e}")
            return None
        
        # 检查空间坐标
        if 'spatial' not in self.sp.obsm:
            return None
        
        # 准备数据
        rna_df = pd.DataFrame(self.rna.X.toarray() if hasattr(self.rna.X, 'toarray') else np.asarray(self.rna.X),
                              index=self.rna.obs_names, columns=self.rna.var_names)
        sp_df = pd.DataFrame(self.sp.X.toarray() if hasattr(self.sp.X, 'toarray') else np.asarray(self.sp.X),
                             index=self.sp.obs_names, columns=self.sp.var_names)
        
        # 选择高变基因
        dge = rna_df.values
        hvg = np.argsort(np.divide(np.var(dge, axis=0), np.mean(dge, axis=0) + 1e-4))
        dge_hvg = dge[:, hvg[-2000:]] if dge.shape[1] >= 2000 else dge
        
        # 设置空间分布和成本
        pts = np.asarray(self.sp.obsm['spatial'])
        p_location, p_expression = nc.rc.create_space_distributions(pts.shape[0], dge_hvg.shape[0])
        cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg, pts, num_neighbors_source=5, num_neighbors_target=5)
        
        # 检查基因可用性
        feature = [g for g in self.train_genes if g in sp_df.columns]
        test = [g for g in self.test_genes if g in sp_df.columns]
        if len(feature) == 0 or len(test) == 0:
            return None
        
        # 计算标记基因成本
        # 确保只使用在两个数据集中都存在的基因
        common_markers = [g for g in feature if g in rna_df.columns]
        if len(common_markers) == 0:
            return None
        
        insitu_matrix = np.array(sp_df[common_markers])
        gene_names = np.array(rna_df.columns.values)
        markers_in_sc = [np.where(gene_names == marker)[0][0] for marker in common_markers]
        
        # 确保维度匹配
        if len(markers_in_sc) != len(common_markers):
            return None
        
        # 标准化数据
        dge_markers = dge[:, markers_in_sc]
        dge_markers_norm = dge_markers / (np.amax(dge_markers) + 1e-8)
        insitu_matrix_norm = insitu_matrix / (np.amax(insitu_matrix) + 1e-8)
        
        cost_marker_genes = cdist(dge_markers_norm, insitu_matrix_norm)
        
        # 运行Gromov-Wasserstein
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,
                                                                0.5, p_expression, p_location, 'square_loss', epsilon=5e-3, verbose=False)
        
        # 计算推断结果
        sdge = np.dot(dge.T, gw)
        imputed = pd.DataFrame(sdge, index=rna_df.columns)
        result = imputed.loc[test].T
        result.index = sp_df.index
        
        return result

    def spaotsc(self) -> Optional[pd.DataFrame]:
        try:
            from spaotsc import SpaOTsc
            from scipy.spatial.distance import cdist
        except Exception as e:
            print(f"SpaOTsc导入失败: {e}")
            return None
        
        # 检查空间坐标
        if 'spatial' not in self.sp.obsm:
            return None
        
        # 准备数据
        df_sc = pd.DataFrame(self.rna.X.toarray() if hasattr(self.rna.X, 'toarray') else np.asarray(self.rna.X),
                             index=self.rna.obs_names, columns=self.rna.var_names)
        df_IS = pd.DataFrame(self.sp.X.toarray() if hasattr(self.sp.X, 'toarray') else np.asarray(self.sp.X),
                             index=self.sp.obs_names, columns=self.sp.var_names)
        
        # 计算空间距离矩阵
        pts = np.asarray(self.sp.obsm['spatial'])
        is_dmat = cdist(pts, pts)
        
        # 选择训练基因
        df_is = df_IS.loc[:, [g for g in self.train_genes if g in df_IS.columns]]
        gene_overlap = list(set(df_is.columns).intersection(df_sc.columns))
        if len(gene_overlap) == 0:
            return None
        
        # 计算相关性矩阵
        a = df_is[gene_overlap]
        b = df_sc[gene_overlap]
        
        # 使用spearmanr计算相关性，与参考代码保持一致
        from scipy import stats
        rho, pval = stats.spearmanr(a, b, axis=1)
        rho[np.isnan(rho)] = 0
        
        # 提取mcc矩阵，与参考代码保持一致
        mcc = rho[-(len(df_sc)):, 0:len(df_is)]
        C = np.exp(1 - mcc)
        
        # 运行SpaOTsc
        issc = SpaOTsc.spatial_sc(sc_data=df_sc, is_data=df_is, is_dmat=is_dmat)
        issc.transport_plan(C ** 2, alpha=0, rho=1.0, epsilon=0.1, cor_matrix=mcc, scaling=False)
        
        # 计算推断结果
        gamma = issc.gamma_mapping
        for j in range(gamma.shape[1]):
            gamma[:, j] = gamma[:, j] / np.sum(gamma[:, j])
        
        X_pred = np.matmul(gamma.T, np.array(issc.sc_data.values))
        result = pd.DataFrame(data=X_pred, columns=issc.sc_data.columns.values, index=self.sp.obs_names)
        
        test = [g for g in self.test_genes if g in result.columns]
        return result.loc[:, test] if test else None

    def stplus(self) -> Optional[pd.DataFrame]:
        try:
            from stPlus import stPlus
            import os
            import torch
        except Exception as e:
            print(f"stPlus导入失败: {e}")
            return None
        
        # 设置设备环境变量，强制stPlus使用指定设备
        if self.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.cuda.set_device(-1)  # 禁用CUDA
        else:
            # 确保使用正确的GPU设备
            if 'cuda' in self.device:
                gpu_id = self.device.split(':')[1] if ':' in self.device else '0'
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        
        # 准备数据
        RNA_data = pd.DataFrame(self.rna.X.toarray() if hasattr(self.rna.X, 'toarray') else np.asarray(self.rna.X),
                                index=self.rna.obs_names, columns=self.rna.var_names).T
        Spatial_data = pd.DataFrame(self.sp.X.toarray() if hasattr(self.sp.X, 'toarray') else np.asarray(self.sp.X),
                                    index=self.sp.obs_names, columns=self.sp.var_names)
        
        # 设置保存路径
        save_path_prefix = None
        if self.outdir:
            os.makedirs(os.path.join(self.outdir, "process_file"), exist_ok=True)
            save_path_prefix = os.path.join(self.outdir, 'process_file/stPlus-demo')
        
        # 运行stPlus
        train_genes_available = [g for g in self.train_genes if g in Spatial_data.columns]
        if not train_genes_available:
            return None
        
        return stPlus(Spatial_data[train_genes_available], RNA_data.T, self.test_genes, save_path_prefix)

    def scfugw(self) -> Optional[pd.DataFrame]:
        try:
            import scFUGW
            import torch
            from scipy.sparse import csr_matrix, issparse
        except Exception:
            return None
        
        # 设置设备
        dev = torch.device(self.device) if self.device else (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))


        # 复制数据
        rna = self.rna.copy()
        sp = self.sp.copy()
        
        # 只保留train基因的spatial数据
        sp_train = sp[:, sp.var_names.isin(self.train_genes)].copy()
        common_genes = list(set(rna.var_names).intersection(sp_train.var_names))

        if len(common_genes) == 0:
            return None

        # 确保X矩阵是CSR格式
        if issparse(sp_train.X) and not isinstance(sp_train.X, csr_matrix):
            sp_train.X = csr_matrix(sp_train.X)

        # 构建 shared space
        rna.obsm['shareSpace'], sp_train.obsm['shareSpace'] = scFUGW.find_shared_space(
            common_genes, rna, sp_train, random_seed=3407, control=None
        )

        # 源域（RNA）cost
        C_source_t = scFUGW.umap_cost(rna, key_added='UMAP4Distance', n_components=2, l2_normalize=True, store_in_obsp=False)
        C_source = C_source_t.cpu().numpy() if hasattr(C_source_t, 'cpu') else np.asarray(C_source_t)
        rna.obsp['cost_matrix'] = torch.tensor(C_source, dtype=torch.float32, device=dev)

        # 为spatial数据生成降维表示
        sp_train = scFUGW.reduction_rna(sp_train, sample_id=None, random_seed=3407)

        # 目标域（Spot）cost
        C_target = scFUGW.knn_cosine_distance(sp_train, obsm_key='RNA_pca_l2_norm', n_neighbors=20, fill_value=1.0)
        sp_train.obsp['cost_matrix'] = torch.tensor(C_target, dtype=torch.float32, device=dev)

        # 特征 cost（M）
        M_t = scFUGW.cosine_distance_tensor(rna.obsm['shareSpace'], sp_train.obsm['shareSpace'])
        M = M_t.cpu().numpy() if hasattr(M_t, 'cpu') else np.asarray(M_t)
        M = torch.tensor(M, dtype=torch.float32, device=dev)

        # 拉普拉斯正则（L）
        if 'spatial' in sp_train.obsm:
            from scFUGW.cost import compute_knn_normalized_laplacian
            L_t = compute_knn_normalized_laplacian(sp_train.obsm['spatial'], n_neighbors=20, sigma=1.0)
            L = L_t.cpu().numpy() if hasattr(L_t, 'cpu') else np.asarray(L_t)
            L = torch.tensor(L, dtype=torch.float32, device=dev)
        else:
            L = None

        # 运行 scFUGW
        pi = scFUGW.scFUGW_RNA_Spatial_with_cost(
            sp_train,
            rna,
            target_cost='cost_matrix',
            source_cost='cost_matrix',
            M=M,
            L=L,
            alpha=0.6,
            rho=0.3,
            eps=0.2,
            lambda_laplacian=5.0,
        )

        # 推断测试基因
        test_rna = rna[:, self.test_genes]
        counts = csr_matrix(test_rna.X)
        pi_np = pi.cpu().numpy() if hasattr(pi, 'cpu') else np.asarray(pi)
        result = counts.T.dot(pi_np)  # (genes x spots)

        imputed = result.T  # (spots x genes)

        # 包装为 DataFrame
        gene_names = list(self.test_genes)
        spot_index = list(sp.obs_names)
        imputed_df = pd.DataFrame(imputed, index=spot_index, columns=gene_names)

        return imputed_df

    def run_single_method(self, method: str) -> Optional[pd.DataFrame]:
        """运行单个指定的方法"""
        method_map = {
            'spage': self.spage,
            'tangram': self.tangram,
            'gimvi': self.gimvi,
            'novosparc': self.novosparc,
            'spaotsc': self.spaotsc,
            'stplus': self.stplus,
            'scfugw': self.scfugw
        }
        
        method_key = method.lower()
        if method_key in method_map:
            return method_map[method_key]()
        else:
            print(f"未知方法: {method}")
            return None

    def run_all(self, methods: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """运行所有指定的方法"""
        method_map = {
            'spage': self.spage,
            'tangram': self.tangram,
            'gimvi': self.gimvi,
            'novosparc': self.novosparc,
            'spaotsc': self.spaotsc,
            'stplus': self.stplus,
            'scfugw': self.scfugw
        }
        
        out = {}
        for method in methods:
            key = method.lower()
            if key in method_map:
                try:
                    out[method] = method_map[key]()
                except Exception as e:
                    print(f"方法 {method} 执行失败: {e}")
                    out[method] = None
            else:
                out[method] = None
        
        return out


# 评估函数
def PCC(raw, impute):
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                pearsonr = 0
            else:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                pearsonr, _ = st.pearsonr(raw_col,impute_col)
            pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[label])
            result = pd.concat([result, pearson_df],axis=1)
    else:
        print("columns error")
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def JS(raw, impute, scale='scale_plus'):
    if scale == 'scale_plus':
        raw = scale_plus(raw)
        impute = scale_plus(impute)
    else:
        print('Please note you do not scale data by plus')

    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                JS = 1
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                raw_col = raw_col.fillna(1e-20)
                impute_col = impute_col.fillna(1e-20)
                M = (raw_col + impute_col) / 2
                JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
            JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
            result = pd.concat([result, JS_df], axis=1)
    else:
        print("columns error")
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result


def RMSE(raw, impute, scale = 'zscore'):
    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print ('Please note you do not scale data by zscore')
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                RMSE = 1.5
            else:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
            result = pd.concat([result, RMSE_df],axis=1)
    else:
        print("columns error")
    return result


def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def SSIM(raw, impute, scale='scale_max'):
    if scale == 'scale_max':
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print('Please note you do not scale data by scale max')
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                ssim = 0
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                M = max(raw_col.max(), impute_col.max())
                raw_col_2 = np.array(raw_col)
                raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                impute_col_2 = np.array(impute_col)
                impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                ssim = cal_ssim(raw_col_2, impute_col_2, M)

            ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
            result = pd.concat([result, ssim_df], axis=1)
    else:
        print("columns error")
    return result


def compute_all_evaluation(raw, impute):
    pcc_result = PCC(raw, impute)
    js_result = JS(raw, impute)
    zscore_result = RMSE(raw, impute)
    SSIM_result = SSIM(raw, impute)

    print('PCC: ', pcc_result.mean().mean())
    print('JS: ', js_result.mean().mean())
    print('RMSE: ', zscore_result.mean().mean())
    print('SSIM: ', SSIM_result.mean().mean())


def evaluate_method(method_name, imputed_result, real_result):
    """
    评估方法性能
    """
    print(f"\n{'='*20} {method_name} 评估结果 {'='*20}")
    
    # 确保基因对齐：只使用共同存在的基因
    common_genes = list(set(imputed_result.columns) & set(real_result.columns))
    print(f"共同基因数量: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print("错误：没有共同基因，无法进行评估")
        return None
    
    # 确保spots对齐：只使用共同存在的spots
    common_spots = list(set(imputed_result.index) & set(real_result.index))
    print(f"共同spots数量: {len(common_spots)}")
    
    if len(common_spots) == 0:
        print("错误：没有共同spots，无法进行评估")
        return None
    
    # 重新对齐数据
    imputed_aligned = imputed_result.loc[common_spots, common_genes]
    real_aligned = real_result.loc[common_spots, common_genes]
    
    print(f"对齐后推断结果形状: {imputed_aligned.shape}")
    print(f"对齐后真实结果形状: {real_aligned.shape}")
    
    # 计算四个评估指标
    print("\n评估指标:")
    
    # 计算PCC
    pcc_result = PCC(real_aligned, imputed_aligned)
    pcc_mean = pcc_result.mean().mean()
    print(f"PCC: {pcc_mean:.4f}")
    
    # 计算JS
    js_result = JS(real_aligned, imputed_aligned)
    js_mean = js_result.mean().mean()
    print(f"JS: {js_mean:.4f}")
    
    # 计算RMSE
    rmse_result = RMSE(real_aligned, imputed_aligned)
    rmse_mean = rmse_result.mean().mean()
    print(f"RMSE: {rmse_mean:.4f}")
    
    # 计算SSIM
    ssim_result = SSIM(real_aligned, imputed_aligned)
    ssim_mean = ssim_result.mean().mean()
    print(f"SSIM: {ssim_mean:.4f}")
    
    return {
        'PCC': pcc_mean,
        'JS': js_mean,
        'RMSE': rmse_mean,
        'SSIM': ssim_mean
    }
