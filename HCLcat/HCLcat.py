#!/usr/bin/env python

import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA, TruncatedSVD
import random
import pandas as pd


def remove_outliers(x, std_limit=5):
    size = len(x)
    std = np.std(x)
    mean = np.mean(x)
    tt = np.where(np.abs(x - mean) > std * std_limit)[0]
    for i in tt:
        x[i] = (x[i + 1] + x[i - 1]) / 2
    return x


def doble_gaussian(x, a, mu, sigma, a2, mu2, sigma2):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + a2 * np.exp(
        -((x - mu2) ** 2) / (2 * sigma2 ** 2)
    )


def prom_mov(x, mov, med=False):
    size = len(x)
    x_mov = np.zeros(size - mov)
    for i in range(size - mov):
        if med:
            x_mov[i] = np.median(x[i : i + mov])
        else:
            x_mov[i] = np.mean(x[i : i + mov])
    return x_mov


def load_doble_gaussian_parameters(snr=10):
    a = 2862.316781326982
    mu = 0.9976487554779495
    sigma = 0.1018979580252512
    a2 = 5417.280508101456
    mu2 = 0.9932463577214291
    sigma2 = -0.040626701263867235
    return a, mu, sigma, a2, mu2, sigma2


def doble_gaussian_noise(snr, lenght=144000):
    a, mu, sigma, a2, mu2, sigma2 = load_doble_gaussian_parameters()
    x = np.linspace(0.1, 2, lenght)
    y = doble_gaussian(x, a, mu, sigma, a2, mu2, sigma2)
    cumulative_distribution_normalized = np.cumsum(y) / np.max(np.cumsum(y))
    inverse_cumulative_distribution = interpolate.interp1d(cumulative_distribution_normalized, x)
    uniform_noise = np.random.uniform(0, 1, lenght)
    return set_snr_level(inverse_cumulative_distribution(uniform_noise), snr)


def normalize(x):
    return x / np.mean(x)


def time_array(x, freq=20):
    size = len(x)
    t = size / (freq * 60)
    return np.linspace(0, t, size)


def set_snr_level(x, snr):
    x = normalize(x)
    signal_snr = calculate_snr(x)
    return (x - 1) * (1 / (snr / signal_snr)) + 1


def calculate_snr(x):
    return np.mean(x) / np.std(x)


def generate_noise_curve(snr, lenght=144000, envelope_level=1):
    ruido = doble_gaussian_noise(snr, lenght * 2)
    freq_ruido = np.fft.fftfreq(lenght * 2, d=0.05)
    fft_ruido = np.fft.fft(ruido)
    expo = generate_exponential_enveloping(freq_ruido, envelope_level=envelope_level)
    ruido = np.fft.ifft(fft_ruido.real * expo)
    return set_snr_level(ruido.real[:lenght], snr)


def exponential_enveloping(x, base, a, b, a2, b2):
    return a * np.exp(-x / b) + a2 * np.exp(-x / b2) + base


def load_exponential_enveloping_parameters():
    base = 1
    a = 4.09589900
    b = 15.55364026e-02
    a2 = 20
    b2 = 8.90201637e-03
    return base, a, b, a2, b2


def generate_exponential_enveloping(fft_freq_array, envelope_level=1):
    size = len(fft_freq_array)
    base, a, b, a2, b2 = load_exponential_enveloping_parameters()
    expo = np.zeros(size)
    for i in range(size):
        if fft_freq_array[i] >= 0:
            expo[i] = exponential_enveloping(
                fft_freq_array[i], base, a * envelope_level, b, a2 * envelope_level, b2
            )
        else:
            expo[i] = exponential_enveloping(
                -fft_freq_array[i], base, a * envelope_level, b, a2 * envelope_level, b2
            )
    return expo


def fourierfilter(x, cutfreq, freq=0.05):
    size = len(x)
    fft_freq = np.fft.fftfreq(size, d=freq)
    x_fft = np.fft.fft(x)
    mask = np.abs(fft_freq) > cutfreq
    x_fft[mask] = 0.1
    x_fft = np.fft.ifft(x_fft)
    return x_fft.real


def pca_train(data_train):
    svd = TruncatedSVD(n_components=15, n_iter=7, random_state=45, tol=0)
    svd.fit(data_train.T)
    comp = svd.components_
    return comp


def pca_filter(data_noised, comp):
    ec_n = np.dot(comp, data_noised)
    return np.matmul(ec_n.T, comp)


def medio_trapezoid(x, depth, down, tc):
    y = np.zeros(len(x))
    a = (depth - 1) / (tc)
    y[: int(down)] = depth
    y[int(down) : int(tc + down)] = -a * (x[int(down) : int(tc + down)] - tc - down) + 1
    y[int(tc + down) :] = 1
    return y


def medio_trapezoid2(x, depth, down, tc, arriba):
    y = np.zeros(len(x))
    a = (depth - 1) / (tc)
    y[: int(down)] = depth
    y[int(down) : int(tc + down)] = -a * (x[int(down) : int(tc + down)] - tc - down) + arriba
    y[int(tc + down) :] = arriba
    return y
