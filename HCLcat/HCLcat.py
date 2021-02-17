#!/usr/bin/env python

from scipy import interpolate
from sklearn.decomposition import PCA, TruncatedSVD
from lmfit import Model

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

SVD_kwargs = dict(n_components=15, n_iter=7, random_state=45, tol=0)

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


def low_pass_fourier_filter(x, cutfreq, freq=0.05):
    size = len(x)
    fft_freq = np.fft.fftfreq(size, d=freq)
    x_fft = np.fft.fft(x)
    mask = np.abs(fft_freq) > cutfreq
    x_fft[mask] = 0
    x_fft = np.fft.ifft(x_fft)
    return x_fft.real


def pca_train(data_train, kwargs=SVD_kwargs):
    svd = TruncatedSVD(**kwargs)
    svd.fit(data_train.T)
    comp = svd.components_
    return comp


def pca_filter(data_noised, comp):
    ec_n = np.dot(comp, data_noised)
    return np.matmul(ec_n.T, comp)


def half_trapezoid(x, depth, down, tc):
    y = np.zeros(len(x))
    a = (depth - 1) / (tc)
    y[: int(down)] = depth
    y[int(down):int(tc + down)] = -a * (x[int(down) : int(tc + down)] - tc - down) + 1
    y[int(tc + down) :] = 1
    return y


def medio_trapezoid2(x, depth, down, tc, arriba):
    y = np.zeros(len(x))
    a = (depth - 1) / (tc)
    y[: int(down)] = depth
    y[int(down) : int(tc + down)] = -a * (x[int(down) : int(tc + down)] - tc - down) + arriba
    y[int(tc + down) :] = arriba
    return y

def initialize_half_trapezoid_model():
    tmod = Model(half_trapezoid)
    tmod.set_param_hint('depth', value=0.9, min=0, max=1)
    tmod.set_param_hint('down', value=5000, min=0, max=144000)
    tmod.set_param_hint('tc', value=18000, max=73000)
    return tmod 

def fit_half_trapezoid_model(data):
    lenght =len(data)
    xdata = np.linspace(1,lenght,lenght)
    model = initialize_half_trapezoid_model()
    params = model.make_params()
    result = model.fit(data, params, x=xdata,method='nelder', max_nfev=20000)
    return result

class TransitLightCurve:
    def __init__(self, data):
        self.data = data
        self.snr = calculate_snr(data)
        self.lenght = len(data)
        self.methods = ["fourier","pca","mov"]
        self.data_filtered = {method : [] for method in self.methods}
        self.cadence = 0.05 #in seconds
        self.data_train = None
        self.pca_comp = None
        self.trap_fitted = {method : [] for method in self.methods}


    def low_pass_filter(self, cutfreq=0.02, plot=True):
        self.data_filtered['fourier'] = low_pass_fourier_filter(self.data, cutfreq, self.cadence)
        if plot:
            plt.plot(self.data_filtered['fourier'])
    
    def mov_filter(self, mov=2000, plot=True, med=False):
        self.data_filtered["mov"] = prom_mov(self.data, mov, med=med)
        if plot:
            plt.plot(self.data_filtered['mov'])
    
    def PCA_train(self, kwargs=SVD_kwargs):
        if self.data_train is not None:
            self.pca_comp = pca_train(self.data_train, kwargs)
        else:
            print("You need update the data_train database first, use: TransitLightCurve.update_PCA_data_train")

    def update_PCA_data_train(self, train_matrix):
        self.data_train = train_matrix

    def PCA_filter(self, plot=True):
        if self.pca_comp is not None:
            self.data_filtered["pca"] = pca_filter(self.data, self.pca_comp)
            if plot:
                plt.plot(self.data_filtered["pca"])
        else:
            print("You need calculate the principal components first, use: TransitLightCurve.PCA_train")   

    def full_analyses(self, plot=True):
        self.low_pass_filter()
        self.mov_filter()
        self.PCA_filter()
        if plot:
            plt.plot(self.data)
        for method in self.trap_fitted.keys():
            self.trap_fitted[method] = fit_half_trapezoid_model(self.data_filtered[method])
            if plot:
                plt.plot(self.trap_fitted[method].best_fit, label=method)
        plt.legend()


    
