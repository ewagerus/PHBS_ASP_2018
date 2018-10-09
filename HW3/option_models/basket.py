# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:56:58 2017

@author: jaehyuk
"""
import numpy as np
import scipy.stats as ss
from option_models import bsm
#from .bsm import bsm_price
#from .normal import normal_price

def basket_check_args(spot, vol, corr_m, weights):
    
    n = spot.size
    assert( n == vol.size )
    assert( corr_m.shape == (n, n) )
    return None
    
def basket_price_mc_cv(
    strike, spot, vol, weights, texp, cor_m, 
    intr=0.0, divr=0.0, cp_sign=1, n_samples=10000):

    # price1 = MC based on BSM
    rand_st = np.random.get_state() # Store random state first
    price1 = basket_price_mc(
        strike, spot, vol, weights, texp, cor_m,
        intr, divr, cp_sign, True, n_samples)
    
    # restore the state in order to generate the same state
    np.random.set_state(rand_st)  
    
    # compute price2: mc price based on normal model
    # make sure you use the same seed
    price2 = basket_price_mc(
        strike, spot, vol*spot, weights, texp, cor_m,
        intr, divr, cp_sign, False, n_samples)
    
    # compute price3: analytic price based on normal model
    # make sure you use the same seed
    price3 = basket_price_norm_analytic(
        strike, spot, vol*spot, weights, texp, cor_m, intr, divr, cp_sign)
    
    # return two prices: without and with CV
    return [price1, price1 + (price3 - price2)] 
    

def basket_price_mc(
    strike, spot, vol, weights, texp, cor_m,
    intr=0.0, divr=0.0, cp_sign=1, bsm=True, n_samples = 10000):
    
    basket_check_args(spot, vol, cor_m, weights)
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac

    cov_m = vol * cor_m * vol[:,None]
    chol_m = np.linalg.cholesky(cov_m)

    n_assets = spot.size
    znorm_m = np.random.normal(size=(n_assets, n_samples))
    
    if( bsm ) :
        
        prices = spot[:,None] * np.exp(-0.5* np.square(vol[:,None]) *texp + (chol_m @ znorm_m) *np.sqrt(texp))
        
        pass 
    
    else:
        prices = forward[:,None] + np.sqrt(texp) * chol_m @ znorm_m
    
    price_weighted = weights @ prices
    
    price = np.mean( np.fmax(cp_sign*(price_weighted - strike), 0) )
    return disc_fac * price


def basket_price_norm_analytic(
    strike, spot, vol, weights, 
    texp, cor_m, intr=0.0, divr=0.0, cp_sign=1):
    
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    
    forward = spot / disc_fac * div_fac
    
    cov_m = vol * cor_m * vol[:,None]
    
    avg_forward = weights @ np.transpose(forward)
    
    vol = np.sqrt(weights@cov_m@np.transpose(weights))

    vol_std = np.fmax(vol * np.sqrt(texp), 1.0e-16)
    
    d = (avg_forward - strike) / (vol_std)

    ck = (avg_forward - strike) * ss.norm.cdf(d) + vol_std * ss.norm.pdf(d)
    
    price = cp_sign * disc_fac * ck
    
    return price

def spread_price_kirk(strike, spot, vol, texp, corr, intr=0, divr=0, cp_sign=1):
    div_fac = np.exp(-texp*divr)
    disc_fac = np.exp(-texp*intr)
    forward = spot / disc_fac * div_fac
    vol2 = vol[1]*forward[1]/(forward[1]+strike)
    vol_r = np.sqrt(vol[0]**2 + vol2*(vol2 - 2*corr*vol[0]))
    price = disc_fac * bsm.bsm_price(forward[1]+strike, forward[0], vol_r, texp, cp_sign=cp_sign)

    return price
