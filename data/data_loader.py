import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from utils.helpers import get_weights, get_group_weights

def data_synthtarget_synthcen(n_data=30000, dataset_str='Gaussian_linear', x_range=[0,2], is_censor=True, seed=42, n_quantiles=11):
    """
    生成合成目标和合成审查的数据集。

    参数:
    - n_data: 数据样本数
    - dataset_str: 数据集类型
    - x_range: 特征范围
    - is_censor: 是否考虑审查
    - seed: 随机种子
    - n_quantiles: 分位数数量

    返回: 
    - train_df, valid_df, test_df: 训练、验证、测试数据集
    - train_true_q_df, valid_true_q_df, test_true_q_df: 对应的真实分位数
    """
    # 导入数据集处理模块
    from .cqrnn_datasets import get_dataset, generate_data_synthtarget_synthcen
    
    tau_seq = np.linspace(1/(n_quantiles+1), 1, n_quantiles+1)[:-1]

    mydataset = get_dataset(dataset_str)
    
    np.random.seed(seed)
    x_train, tte_train, cen_train, y_train, cen_indicator, obs_indicator = generate_data_synthtarget_synthcen(
        n_data=n_data, mydataset=mydataset, x_range=x_range, is_censor=is_censor
    )

    x_train = np.array(x_train).reshape(n_data, -1)
    tte_train = np.array(tte_train).reshape(n_data, -1)
    cen_train = np.array(cen_train).reshape(n_data, -1)
    y_train = np.array(y_train).reshape(n_data, -1)
    cen_indicator = np.array(cen_indicator).reshape(n_data, -1)
    obs_indicator = np.array(obs_indicator).reshape(n_data, -1)

    dim = x_train.shape[1]
    data_pd = pd.DataFrame(
        np.concatenate([x_train, y_train, obs_indicator], axis=1), 
        columns=['x' + str(i) for i in range(dim)] + ['y', 'obs_indicator']
    )

    # 拆分训练、验证和测试集
    train_valid_df, test_df = train_test_split(data_pd, test_size=1/3, random_state=42)
    train_df, valid_df = train_test_split(train_valid_df, test_size=0.5, random_state=42)

    # 计算训练集真实分位数值
    train_true_q = {}
    for q in tau_seq:
        y_true = mydataset.get_quantile_truth(train_df.loc[:, ~train_df.columns.isin(['y', 'obs_indicator'])].squeeze().to_numpy(), q).reshape(-1, 1)
        formatted_q = f'{q:.4f}'
        train_true_q['q' + str(formatted_q)] = y_true.flatten()
    train_true_q_df = pd.DataFrame(train_true_q)

    # 计算测试集真实分位数值
    test_true_q = {}
    for q in tau_seq:
        y_true = mydataset.get_quantile_truth(test_df.loc[:, ~test_df.columns.isin(['y', 'obs_indicator'])].squeeze().to_numpy(), q).reshape(-1, 1)
        formatted_q = f'{q:.4f}'
        test_true_q['q' + str(formatted_q)] = y_true.flatten()
    test_true_q_df = pd.DataFrame(test_true_q)

    # 计算验证集真实分位数值
    valid_true_q = {}
    for q in tau_seq:
        y_true = mydataset.get_quantile_truth(valid_df.loc[:, ~valid_df.columns.isin(['y', 'obs_indicator'])].squeeze().to_numpy(), q).reshape(-1, 1)
        formatted_q = f'{q:.4f}'
        valid_true_q['q' + str(formatted_q)] = y_true.flatten()
    valid_true_q_df = pd.DataFrame(valid_true_q)

    return train_df, valid_df, test_df, train_true_q_df, valid_true_q_df, test_true_q_df

def organize_data(df, time=["y"], event=["obs_indicator"], trt=None):
    """
    组织数据为模型所需格式。

    参数:
    - df: 数据框
    - time: 时间列
    - event: 事件列
    - trt: 处理列

    返回:
    - 组织好的数据字典
    """
    E = np.array(df[event])
    Y = np.array(df[time])
    print('Y: ', Y.shape)
    X = np.array(df.drop(event + time, axis=1))
    X = X.astype('float64')
    scaler = preprocessing.StandardScaler().fit(X)  # standardize
    X = scaler.transform(X)
    
    if trt is None:
        W = get_weights(Y, E)
        return {"X": X, "Y": Y, "E": E, "W": W}
    else:
        trt = np.array(df[trt])
        W = get_group_weights(Y, E, trt)
        return {"X": X, "Y": Y, "E": E, "W": W, "trt": trt}

def load_custom_dataset(dataset_str, seed=42):
    """
    加载自定义数据集。

    参数:
    - dataset_str: 数据集名称
    - seed: 随机种子

    返回:
    - train_df, valid_df, test_df: 处理后的数据
    """
    def remove_zero_duration(df):
        return df[df['duration'] != 0]
    
    if dataset_str in ['metabric', 'support', 'gbsg', 'gbsg500', 'nki70', 'flchain']:
        # 加载真实数据集
        train_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_train.csv"
        train_df = pd.read_csv(train_dataset_fp)
        train_df = remove_zero_duration(train_df)
        train_df = organize_data(train_df, time=["duration"], event=["event"])

        valid_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_val.csv"
        valid_df = pd.read_csv(valid_dataset_fp)
        valid_df = remove_zero_duration(valid_df)
        valid_df = organize_data(valid_df, time=["duration"], event=["event"])

        test_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_test.csv"
        test_df = pd.read_csv(test_dataset_fp)
        test_df = remove_zero_duration(test_df)
        test_df = organize_data(test_df, time=["duration"], event=["event"])
    else:
        # 默认数据集路径
        train_dataset_fp = "./data/traindata.csv"
        train_df = pd.read_csv(train_dataset_fp)
        train_df = organize_data(train_df, time=["OT"], event=["ind"], trt=["x2"])

        valid_dataset_fp = "./data/testdata.csv"
        valid_df = pd.read_csv(valid_dataset_fp)
        valid_df = organize_data(valid_df, time=["OT"], event=["ind"], trt=["x2"])

        test_dataset_fp = "./data/testdata.csv"
        test_df = pd.read_csv(test_dataset_fp)
        test_df = organize_data(test_df, time=["OT"], event=["ind"], trt=["x2"])

    return train_df, valid_df, test_df

def load_multi_diseases_dataset(dataset_str, seed=42):
    """
    加载多疾病数据集。

    参数:
    - dataset_str: 数据集名称
    - seed: 随机种子

    返回:
    - train_df, valid_df, test_df: 处理后的数据
    """
    def remove_zero_duration(df):
        return df[df['duration'] != 0]
    
    if dataset_str in ['metabric', 'support', 'gbsg', 'gbsg500', 'nki70', 'flchain']:
        train_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_train.csv"
        train_df = pd.read_csv(train_dataset_fp)
        train_df = remove_zero_duration(train_df)
        train_df = organize_data(train_df, time=["duration","duration","duration"], event=["event","event","event"])

        valid_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_val.csv"
        valid_df = pd.read_csv(valid_dataset_fp)
        valid_df = remove_zero_duration(valid_df)
        valid_df = organize_data(valid_df, time=["duration","duration","duration"], event=["event","event","event"])

        test_dataset_fp = f"/work/users/s/h/shuaishu/Intern/Servier/real_data/{dataset_str}/{dataset_str}_seed{seed}_test.csv"
        test_df = pd.read_csv(test_dataset_fp)
        test_df = remove_zero_duration(test_df)
        test_df = organize_data(test_df, time=["duration","duration","duration"], event=["event","event","event"])
    
    return train_df, valid_df, test_df

def calculate_true_quantiles(quantiles):
    """
    计算每个指数分布的分位数的组合分布。

    参数:
    - quantiles: 分位数列表

    返回:
    - 组合分位数值
    """
    # Rate parameters λ
    lambda_series = np.array([1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.1, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 1.8, 2])
    
    # Calculate quantiles for each exponential distribution
    quantile_values = np.array([[-np.log(1-p) / lam for p in quantiles] for lam in lambda_series])
    
    # 计算T的组合分布分位数
    combined_quantiles = np.mean(quantile_values, axis=0)
    
    return combined_quantiles

def generate_true_quantile_data(n, quantiles):
    """
    为每个观测值生成真实分位数数据。

    参数:
    - n: 观测个数
    - quantiles: 分位数列表

    返回:
    - DataFrame包含真实分位数
    """
    # 计算真实分位数
    true_quantiles = calculate_true_quantiles(quantiles)
    
    # 为每个观测值创建包含真实分位数的DataFrame
    true_quantile_data = pd.DataFrame(
        np.tile(true_quantiles, (n, 1)), 
        columns=[f"Q{int(q*100)}" for q in quantiles]
    )
    
    return true_quantile_data