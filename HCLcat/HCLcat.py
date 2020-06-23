#!/usr/bin/env python

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA, TruncatedSVD, SparseCoder, NMF
from sklearn.decomposition import MiniBatchDictionaryLearning, KernelPCA
import random
import pandas as pd

def doble_gaussian (x,a,mu,sigma,a2,mu2,sigma2):
    return a*np.exp(-(x - mu)**2 / (2 * sigma**2)) + a2*np.exp(-(x - mu2)**2 / (2 * sigma2**2))

def gaussiana(x,a,mu,sigma):
    return a*np.exp(-(x - mu)**2 / (2 * sigma**2))

def snr(x):
    return (np.mean(x))/(np.std(x))

def red(noise,lenght=144000):
    output = np.ones(lenght)
    for i in range(lenght - 1):
        output[i]=(0.5 * (noise[i] + noise[i+1]))
    return output

def genruidorojo(noise,lenght=144000): 
    x=genruido(noise,lenght)
    x2=red(x,lenght)
    x=(x+x2)
    return set_noise_level(x,noise)
    
    
def set_noise_level(curve,noise):
    curve=curve/np.mean(curve)
    signal_noise=snr(x)
    curve=curve-1
    curve=curve*(1/(noise/signal_noise))
    return curve+1
    
def genruido(noise,lenght=288000): 
    a=2862.316781326982 
    mu=0.9976487554779495 
    sigma=0.1018979580252512
    a2=5417.280508101456 
    mu2=0.9932463577214291 
    sigma2=-0.040626701263867235
    x=np.linspace(0.1,2,lenght)
    y=doble_gaussian(x,a,mu,sigma,a2,mu2,sigma2)
    cumulative_distribution = np.cumsum(y)
    cumulative_distribution_normalized=C/np.max(cumulative_distribution)
    inverse_cumulative_distribution = interpolate.interp1d(cumulative_distribution_normalized,x)
    uniform_noise = np.random.uniform(0,1,lenght)
    ruido=inverse_cumulative_distribution(uniform_noise)
    return set_noise_level(ruido,noise)   

def remove_outliers(x,std_limit=5):
    size=x.shape[0]
    std=np.std(x)
    mean=np.mean(x)
    tt = np.where(np.abs(x-mean) > std*std_limit)[0]
    for i in tt:
        x[i]=(x[i+1]+x[i-1])/2
    return x

def prom_mov(x,mov,method='prom'):
	size=x.shape[0]
	x_mov=np.zeros(size-mov)
	for i in range(size-mov):
        if method == 'prom':
		    x_mov[i]=np.sum(x[i:i+mov])/mov
        if method == 'med':
            x_mov[i]=np.median(x[i:i+mov])
	return x_mov

def tiempo(x,freq=20):
	size=x.shape[0]
	t=size/(freq*60)
	return np.linspace(0,t,size)

def dexp(x,base,a,b,a2,b2):
    expo=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]>= 0:
            expo[i]=base+a*np.exp(-x[i]/b)+a2*np.exp(-x[i]/b2)
        else:
            expo[i]=base+a*np.exp(x[i]/b)+a2*np.exp(x[i]/b2)
    return expo
            
    
def ruido_real(noise,base,a,b,a2,b2,lenght=288000):
    ruido=genruido(noise,lenght)
    freq_ruido = np.fft.fftfreq(lenght,d=0.05)
    fft_ruido=np.fft.fft(ruido)
    expo=dexp(freq_ruido,base,a,b,a2,b2)
    ruido=np.fft.ifft(fft_ruido.real*expo)
    return set_noise_level(ruido.real,noise) 

def fourierfilter(x,cutfreq):
	size=x.shape[0]
	freq = np.fft.fftfreq(size,d=0.05)
	x_fft=np.fft.fft(x)
	mask = np.abs(freq) > cutfreq
	x_fft[mask]=0.1    
	x_fft=np.fft.ifft(x_fft)
	return x_fft

def pca_train(data_train):
    svd = TruncatedSVD(n_components=15, n_iter=7,random_state=45,tol=0)
    svd.fit(data_train.T)
    comp=svd.components_
    return comp

def pca_filter(data_noised,comp):
    ec_n=np.dot(comp,data_noised)  
    return np.matmul(ec_n.T,comp)

def medio_trapezoid(x,depth,down,tc):
    y = np.zeros(len(x))
    a = ((depth-1)/(tc))
    y[:int(down)]=depth
    y[int(down):int(tc+down)] =  -a*(x[int(down):int(tc+down)]-tc-down) + 1
    y[int(tc+down):]=1
    return y

def medio_trapezoid2(x,depth,down,tc,arriba):
    y = np.zeros(len(x))
    a = ((depth-1)/(tc))
    y[:int(down)]=depth
    y[int(down):int(tc+down)] =  -a*(x[int(down):int(tc+down)]-tc-down) + arriba
    y[int(tc+down):]=arriba
    return y