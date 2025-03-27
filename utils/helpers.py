import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import KaplanMeierFitter
from sklearn import preprocessing

def get_weights(time, delta):
    """
    Compute inverse probability censoring weights for multiple diseases.

    Parameters:
    - time: np.array of shape (batch, D), observed failure times for each disease.
    - delta: np.array of shape (batch, D), censoring indicators (1 = event, 0 = censored).

    Returns:
    - w: np.array of shape (batch, D), calculated weights for each disease.
    """
    batch_size, num_diseases = time.shape  # Ensure time is (batch, D)
    w = np.zeros((batch_size, num_diseases))  # Initialize weight matrix

    for d in range(num_diseases):
        kmf = KaplanMeierFitter()
        
        # Fit KM estimator for each disease independently
        kmf.fit(durations=time[:, d], event_observed=1 - delta[:, d])  

        # Compute survival function estimate for each time point
        km = np.array(kmf.predict(time[:, d]))  

        # Avoid zero division by replacing zeros with a small positive value
        km[km == 0] = 0.005  

        # Compute weights for each disease
        w[:, d] = delta[:, d] / km  

    return w

def get_group_weights(time, delta, trt):
    n = np.shape(time)[0]
    w = np.zeros((n,))
    w1 = get_weights(time[trt==0], delta[trt==0])
    w2 = get_weights(time[trt==1], delta[trt==1])
    w[trt==0] = w1
    w[trt==1] = w2
    return w

def huber(y_true, y_pred, eps=0.001):
    error = y_true - y_pred
    cond = torch.abs(error) < eps

    squared_loss = (error ** 2) / (2 * eps)
    linear_loss = torch.abs(error) - 0.5 * eps

    return torch.where(cond, squared_loss, linear_loss)

def weighted_loss(weights, tau, eps=0.001):
    def loss(y_true, y_pred):
        e = huber(y_true, y_pred)
        e = weights * e
        return torch.mean(torch.max(tau * e, (tau - 1) * e))
    return loss

class WeightedLoss(nn.Module):
    def __init__(self, tau_seq, eps=0.001):
        super(WeightedLoss, self).__init__()
        self.tau_seq = tau_seq
        self.eps = eps

    def forward(self, y_true, y_pred, weights):
        total_loss = 0
        for i, tau in enumerate(self.tau_seq):
            e = huber(y_true, y_pred[:,i], self.eps)
            e = weights * e
            loss = torch.mean(torch.max(tau * e, (tau - 1) * e))
            total_loss += loss
        return total_loss / len(self.tau_seq)

class WeightedLoss2(nn.Module):
    def __init__(self, tau_seq, eps=0.001):
        """
        加权分位数损失，支持多个疾病 (D diseases) 和多个分位数 (K quantiles)

        参数:
        tau_seq: 一个包含分位数值的列表 (如 [0.1, 0.5, 0.9])
        eps: 避免数值问题的小量 (默认为 0.001)
        """
        super(WeightedLoss2, self).__init__()
        self.tau_seq = tau_seq
        self.eps = eps

    def forward(self, y_true, y_pred, weights):
        """
        计算加权分位数损失
        
        参数:
        y_true: 真实值, 形状 (batch_size, D)
        y_pred: 预测值, 形状 (batch_size, D, K)
        weights: 权重, 形状 (batch_size, D)

        返回:
        加权分位数损失 (scalar)
        """
        batch_size, num_diseases, num_quantiles = y_pred.shape
        print('y_pred:', y_pred.shape)

        # 调整 y_true 形状，使其匹配 y_pred
        y_true = y_true.unsqueeze(-1)  # (batch_size, D) -> (batch_size, D, 1)

        # **扩展 weights，使其适用于 K 维度**
        weights = weights.unsqueeze(-1)  # (batch_size, D) -> (batch_size, D, 1)

        # 计算误差 (batch_size, D, K)
        diff_y = y_true - y_pred  # 计算 (y_true - y_pred) 形状 (batch_size, D, K)

        # 计算分位数损失
        sum_loss = weights * diff_y * (torch.tensor(self.tau_seq, device=y_pred.device) - (torch.sign(-diff_y) + 1) / 2)

        # 计算所有 quantile 的均值损失
        return sum_loss.mean()


def organize_data(df, time=["OT"], event=["ind"], trt=None):
    E = np.array(df[event])
    Y = np.array(df[time])
    print('Y:', Y.shape)
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

def kernel_univariate(x0, x, h):
    """
    Calculate univariate biquadratic kernel weights using PyTorch.
    
    Parameters:
    -----------
    x0 : torch.Tensor
        Point at which to calculate weights
    x : torch.Tensor
        Vector of observed covariate values for one dimension
    h : float
        Bandwidth parameter
    
    Returns:
    --------
    torch.Tensor
        Vector of kernel weights for this dimension
    """
    xx = (x - x0) / h
    w = torch.zeros_like(xx, device=x.device)
    mask = torch.abs(xx) < 1
    w[mask] = 15 * (1 - xx[mask]**2)**2 / 16
    return w

def Bnk_func(x0, x, h):
    """
    Calculate multivariate kernel weights as product of univariate kernels.
    
    Parameters:
    -----------
    x0 : torch.Tensor
        Point at which to calculate weights (p-dimensional)
    x : torch.Tensor
        Matrix of observed covariates (n x p)
    h : float or torch.Tensor
        Bandwidth parameter(s). If float, same bandwidth for all dimensions.
        If tensor, should have length p for dimension-specific bandwidths.
    
    Returns:
    --------
    torch.Tensor
        Vector of kernel weights (length n)
    """
    n, p = x.shape
    
    # Convert h to tensor if it's scalar
    if isinstance(h, (float, int)):
        h = torch.full((p,), h, device=x.device)
    
    # Initialize weights to ones
    weights = torch.ones(n, device=x.device)
    
    # Calculate product kernel
    for j in range(p):
        weights *= kernel_univariate(x0[j], x[:, j], h[j])
    
    # Normalize weights
    weights = weights / torch.sum(weights)
    return weights

def tauhat_func(y0, x0, z, x, delta, h):
    """
    Generalized Kaplan-Meier estimator for conditional survival probability.
    
    Parameters:
    -----------
    y0 : torch.Tensor
        Time point for estimation
    x0 : torch.Tensor
        Covariate values for estimation (p-dimensional)
    z : torch.Tensor
        Observed response variables
    x : torch.Tensor
        Observed covariates (n x p)
    delta : torch.Tensor
        Censoring indicators (1=uncensored, 0=censored)
    h : float or torch.Tensor
        Bandwidth parameter(s)
    
    Returns:
    --------
    torch.Tensor
        Estimated conditional probability
    """
    n = len(z)
    Bn = Bnk_func(x0, x, h)
    
    if y0 < torch.max(z):
        # Sort data
        sort_idx = torch.argsort(z)
        z2 = z[sort_idx]
        Bn2 = Bn[sort_idx]
        delta2 = delta[sort_idx]
        
        # Find relevant indices
        eta = torch.where((delta2 == 1) & (z2 <= y0))[0]
        
        if len(eta) > 0:
            # Calculate cumulative sum from reversed order
            Bn3 = torch.flip(Bn2, [0])
            cumsum_Bn3 = torch.flip(torch.cumsum(Bn3, 0), [0])
            
            # Calculate temporary values
            tmp = 1 - Bn2 / cumsum_Bn3
            
            # Remove any NA values and calculate product
            valid_tmp = tmp[eta][~torch.isnan(tmp[eta])]
            out = 1 - torch.prod(valid_tmp)
        else:
            out = torch.tensor(0.0, device=z.device)
    else:
        out = torch.tensor(1.0, device=z.device)
        
    return out

def quantile_loss_matrix(residuals, tau):
    """
    Calculate quantile loss for multiple quantiles using broadcasting.
    
    Parameters:
    -----------
    residuals : torch.Tensor
        Matrix of residuals (n x q)
    tau : torch.Tensor
        Vector of quantile levels (q,)
    
    Returns:
    --------
    torch.Tensor
        Matrix of quantile losses (n x q)
    """
    # Ensure tau is a tensor
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, device=residuals.device, dtype=residuals.dtype)
    
    tau = tau.unsqueeze(0)  # (1 x q)
    return torch.where(residuals >= 0,
                      tau * residuals,
                      (tau - 1) * residuals)

def multi_censored_quantile_loss(y, y_pred, x, delta, tau, h=0.05):
    """
    Calculate multi-quantile censored regression loss with multivariate covariates.
    
    Parameters:
    -----------
    y_pred : torch.Tensor
        Predicted values (n x q)
    y : torch.Tensor
        Observed values (n,)
    x : torch.Tensor
        Covariates (n x p)
    delta : torch.Tensor
        Censoring indicators (n,)
    tau : list or torch.Tensor
        Quantile levels (q,)
    h : float or torch.Tensor
        Bandwidth parameter(s)
    
    Returns:
    --------
    torch.Tensor
        Average loss
    """
    # Convert tau to tensor if it's a list
    if isinstance(tau, list):
        tau = torch.tensor(tau, device=y.device, dtype=y.dtype)
    
    n, p = x.shape
    q = len(tau)
    
    if y_pred.shape != (n, q):
        raise ValueError(f"y_pred must have shape ({n}, {q}), got {y_pred.shape}")
    
    # Initialize weights
    weights = torch.ones((n, q), device=y.device)
    
    # Find censored observations
    censored_idx = torch.where(delta == 0)[0]
    
    if len(censored_idx) >= 1:
        # Calculate weights for censored observations
        for i in censored_idx:
            tau_star = tauhat_func(y[i], x[i], y, x, delta, h)
            weights[i] = torch.where(tau > tau_star,
                                   (tau - tau_star) / (1 - tau_star),
                                   torch.tensor(1.0, device=y.device))
        
        # Create pseudo observations
        y_pse = torch.full((n,), torch.max(y) + 100, device=y.device)
    else:
        y_pse = y.clone()
    
    # Calculate losses
    y_expanded = y.unsqueeze(1)      # (n x 1)
    y_pse_expanded = y_pse.unsqueeze(1)  # (n x 1)
    
    original_loss = quantile_loss_matrix(y_expanded - y_pred, tau)
    pseudo_loss = quantile_loss_matrix(y_pse_expanded - y_pred, tau)
    
    # Combine losses using weights
    total_loss = weights * original_loss + (1 - weights) * pseudo_loss
    
    # Calculate average losses
    quantile_losses = torch.mean(total_loss, dim=0)  # (q,)
    average_loss = torch.mean(quantile_losses)        # scalar
    
    return average_loss