# DeAnomaly: Multivariate Time Series Anomaly Detection with Decomposition and Diffusion

**Abstract:** Multivariate time seris anomaly detection (MTS-AD) is of great significance in various modern industrial applications and IT systems.
Recently, some unsupervised deep models have been developed for MTS-AD.
However, these methods often struggle to handle the complex temporal patterns and inevitable noise in multivariate time series (MTS) data, resulting in limited performance.
To overcome these challenges, we propose DeAnomaly, a novel anomaly detection framework based on time series decomposition.
DeAnomaly employs a two-phase training paradigm, consisting of structural pattern elimination and anomaly detection on remainders.
The structural pattern elimination phase learns normal trend and seasonal components through spatial relationship modeling and time-frequency analysis, which are subsequently removed from the original time series to overcome the limitation of complex temporal patterns.
The anomaly detection phase utilizes the robust characteristics of noise with denoising diffusion models to identify and distinguish between noise and actual anomalies.
Since anomalies and small random fluctuations are mainly retained in the remainders, anomalies will be more clearly exposed. In this way, DeAnomaly can detect anomalies more accurately and robustly.
We conduct extensive experiments on four real-world datasets and eleven baselines, experimental results demonstrate that DeAnomaly outperforms these state-of-the-arts. 

## Requirements

- Python 3.9
- PyTorch version 1.13.1+cu117
- numpy
- scipy
- pandas
- Pillow
- scikit-learn
- xlrd

## Dependencies can be installed using the following command:

```
pip install -r requirements.txt
```

## Datasets

MSL, SMAP, and SWaT datasets were acquired at the following link. Put them in the datasets folder.

```
https://drive.google.com/drive/folders/1LUNotax6LGdFOlyThEJp7oCNN2fqNUNt?usp=sharing
```

## Usage

To reproduce the results mentioned in our paper, we provide a complete command for training and testing DeAnomaly:

```
python main.py --num_epochs <num_epochs> --batch_size <batch_size> --mode <mode> --dataset <dataset> --data_path <data_path> --input_c <input_c> --output_c <output_c>
```

