import torch


def apply_random_mask(time_series, mask_ratio=0.15):
    """
    对给定的多变量时间序列应用随机掩码。

    参数:
    - time_series: 一个形状为 (batch_size, feature_dim, time_window) 的张量。
    - mask_ratio: 要掩码的元素比例。

    返回:
    - time_series_masked: 应用了随机掩码的时间序列。
    """
    # 确定时间序列的维度
    batch_size, feature_dim, time_window = time_series.shape

    # 计算总元素数和需要掩码的元素数
    total_elements = batch_size * feature_dim * time_window
    mask_count = int(total_elements * mask_ratio)

    # 生成随机索引
    indices = torch.randperm(total_elements)[:mask_count]

    # 创建初始全为 False 的掩码
    mask = torch.zeros(total_elements, dtype=torch.bool).to("cuda")

    # 设置随机选定位置为 True
    mask[indices] = True

    # 调整掩码形状以匹配时间序列
    mask = mask.view(batch_size, feature_dim, time_window)

    # 应用掩码，设置选定元素为 0
    time_series_masked = time_series.masked_fill(mask, 0)

    return time_series_masked


def generate_mask_for_timeseries(timeseries_data, mask_ratio=0.15):
    """
    生成随机掩码矩阵，用于输入的多变量时间序列的段掩码。

    参数:
    - timeseries_data: torch.Tensor, 时间序列数据，形状为 [batch_size, num_features, time_window]。
    - mask_ratio: float, 掩码比例，默认为15%。

    返回:
    - mask_matrix: torch.Tensor, 掩码矩阵。
    """
    batch_size, num_features, time_window = timeseries_data.shape
    mask_length = int(time_window * mask_ratio)

    # 初始化掩码矩阵为1
    mask_matrix = torch.ones_like(timeseries_data)

    for i in range(batch_size):
        # 随机确定掩码的起始点
        start = torch.randint(0, time_window - mask_length + 1, (1,)).item()
        end = start + mask_length

        # 将掩码部分设置为0
        mask_matrix[i, :, start:end] = 0

    return mask_matrix