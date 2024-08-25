import torch


def generate_mask_for_timeseries(timeseries_data, mask_ratio=0.15):
    batch_size, num_features, time_window = timeseries_data.shape
    mask_length = int(time_window * mask_ratio)
    mask_matrix = torch.ones_like(timeseries_data)

    for i in range(batch_size):
        start = torch.randint(0, time_window - mask_length + 1, (1,)).item()
        end = start + mask_length
        mask_matrix[i, :, start:end] = 0

    return mask_matrix