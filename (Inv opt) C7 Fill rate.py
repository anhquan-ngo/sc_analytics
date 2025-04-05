import numpy as np
import pandas as pd
from scipy.stats import norm


def normal_loss(inv,mu,std):
    return std**2*norm.pdf(inv, mu, std) + (mu - inv)*(1-norm.cdf(inv, mu, std))

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

inv = 120
mu = 100
std = 50
print(normal_loss(inv,mu,std))
print(std*normal_loss_standard((inv-mu)/std))