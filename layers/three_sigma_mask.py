import torch


def generate_mask_for_timeseries(data):
    mean = torch.mean(data, dim=2, keepdim=True)
    std = torch.std(data, dim=2, keepdim=True)

    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    anomaly_mask = (data < lower_bound) | (data > upper_bound)

    masked_data = torch.where(anomaly_mask, torch.zeros_like(data), data)

    return masked_data