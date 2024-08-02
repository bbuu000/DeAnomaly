import torch
import torch.nn as nn

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        if self.kernel_size % 2 == 0:
            front = x[:, 0:1, :].repeat(1, self.kernel_size // 2, 1)
        else:
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class tsr_decomp(nn.Module):
    """
    Series decomposition block (3 parts)
    """
    def __init__(self, kernel_size, period):
        super(tsr_decomp, self).__init__()
        self.series_decomp = series_decomp_multi(kernel_size)
        self.period = period

    def forward(self, x):
        res, trend_init = self.series_decomp(x)
        seasonal = torch.Tensor()
        for j in range(self.period):
            period_average = torch.unsqueeze(torch.mean(res[:, j::self.period, :], axis=1), dim=1)
            seasonal = torch.concat([seasonal, period_average.to('cpu')], dim=1)
        seasonal = seasonal - torch.unsqueeze(torch.mean(seasonal, dim=1), dim=1)
        seasonal_init = torch.tile(seasonal.T, (1, x.shape[1] // self.period + 1, 1)).T[:, :x.shape[1], :]
        resid_init = res - seasonal_init

        return trend_init, seasonal_init, resid_init