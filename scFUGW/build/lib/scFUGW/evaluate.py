import numpy as np
import pandas as pd
import scipy.stats as st



def find_threshold(values, threshold = 1e-3):
    for i, v in enumerate(values):
        if v < threshold:
            print(i, v)
            return i



def get_acc(MP, MT, mappedData, truthData, top=1):
    length = len(MP)
    counts = 0

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        if mappedData[i] == truthData[indexes].value_counts().index[0]:
            counts += 1
    return counts / length



def label_transfer(MP, MT, truthData, top=1):
    predicted_label = []

    for i, point in enumerate(MP):
        distances = np.linalg.norm(MT - point, axis=1)
        indexes = np.argpartition(distances, top)[:top]

        predicted_label.append(truthData[indexes].value_counts().index[0])

    return predicted_label



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
                # [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
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
