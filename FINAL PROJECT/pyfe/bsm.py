import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from .option_model import OptionModelABC
from .rational import BetaFunction


class BsmModel(OptionModelABC):
    sigma, intr, divr = None, 0.0, 0.0
    is_fwd = False
    disc_fac = None

    def __init__(self, sigma, intr=0.0, divr=0.0, is_fwd=False, disc_fac=None):
        self.sigma = sigma
        self.intr = intr
        self.divr = divr
        self.is_fwd = is_fwd
        self.disc_fac = disc_fac

    @staticmethod
    def price_formula(strike, spot_or_fwd, sigma, texp, cp_sign=1, intr=0.0, divr=0.0,
                      is_fwd=False, disc_fac=None):

        disc_fac = disc_fac or np.exp(-texp * intr)
        fwd = spot_or_fwd * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac)

        if texp < 0:
            return disc_fac * np.fmax(cp_sign * (fwd - strike), 0)

        sigma_std = np.fmax(sigma * np.sqrt(texp), 1e-16)
        d1 = np.log(fwd / strike) / sigma_std + 0.5*sigma_std
        d2 = d1 - sigma_std

        price = cp_sign * disc_fac * \
                (fwd * ss.norm.cdf(cp_sign * d1) - strike * ss.norm.cdf(cp_sign * d2))
        return price

    def price(self, strike, spot, texp, cp_sign=1):
        return self.price_formula(
            strike, spot, self.sigma, texp, cp_sign,
            intr=self.intr, divr=self.divr, is_fwd=self.is_fwd, disc_fac=self.disc_fac)

    @staticmethod
    def vega_formula(strike, spot_or_fwd, sigma, texp, cp_sign=1, intr=0.0, divr=0.0,
                     is_fwd=False, disc_fac=None):
        # cp_sign is not used
        disc_fac = disc_fac or np.exp(-texp * intr)
        fwd = spot_or_fwd * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac)

        sigma_std = sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        vega = disc_fac * fwd * ss.norm.pdf(d1) * np.sqrt(texp)  # formula according to wikipedia
        return vega

    def vega(self, strike, spot, texp, cp_sign=1):
        return self.vega_formula(
            strike, spot, self.sigma, texp, cp_sign,
            intr=self.intr, divr=self.divr, is_fwd=self.is_fwd, disc_fac=self.disc_fac)

    @staticmethod
    def delta_formula(strike, spot_or_fwd, sigma, texp, cp_sign=1, intr=0.0, divr=0.0,
                      is_fwd=False, disc_fac=None):
        # cp_sign is not used
        disc_fac = disc_fac or np.exp(-texp * intr)
        div_fac = np.exp(-texp * divr)
        fwd = spot_or_fwd * (1.0 if is_fwd else div_fac/disc_fac)

        sigma_std = sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        delta = ss.norm.cdf(d1)  # formula according to wikipedia
        delta *= div_fac if is_fwd else disc_fac
        return delta

    def delta(self, strike, spot, texp, cp_sign=1):
        return self.delta_formula(
            strike, spot, self.sigma, texp, cp_sign,
            intr=self.intr, divr=self.divr, is_fwd=self.is_fwd, disc_fac=self.disc_fac
        )

    @staticmethod
    def gamma_formula(
            strike, spot_or_fwd, sigma, texp, cp_sign=1, intr=0.0, divr=0.0, is_fwd=False, disc_fac=None):
        # cp_sign is not used
        disc_fac = disc_fac or np.exp(-texp * intr)
        div_fac = np.exp(-texp * divr)
        fwd = spot_or_fwd * (1.0 if is_fwd else div_fac/disc_fac)

        sigma_std = sigma * np.sqrt(texp)
        d1 = np.log(fwd / strike) / sigma_std + 0.5 * sigma_std

        gamma = disc_fac * ss.norm.pdf(d1)/fwd/sigma_std  # formula according to wikipedia
        if not is_fwd:
            gamma *= np.square(div_fac/disc_fac)
        return gamma

    def gamma(self, strike, spot, texp, cp_sign=1):
        return self.gamma_formula(
            strike, spot, self.sigma, texp, cp_sign,
            intr=self.intr, divr=self.divr, is_fwd=self.is_fwd, disc_fac=self.disc_fac
        )

    @staticmethod
    def impvol_base(price, strike, spot_or_fwd, texp, cp_sign=1, intr=0.0, divr=0.0,
               is_fwd=False, disc_fac=None):

        disc_fac = disc_fac or np.exp(-texp * intr)
        fwd = spot_or_fwd * (1.0 if is_fwd else np.exp(-texp * divr) / disc_fac)

        strike_normalized = np.atleast_1d(strike / fwd)   # strike / fwd
        price_normalized = np.atleast_1d(price / disc_fac / fwd)   # forward price / fwd

        n_price = len(price_normalized)
        assert n_price == len(strike_normalized)

        sigma = np.zeros(n_price)
        for k in range(n_price):
            iv_func = lambda _sigma: \
                BsmModel.price_formula(strike_normalized[k], 1.0, _sigma, texp, cp_sign, is_fwd=True, disc_fac=1.0)\
                - price_normalized[k]

            sigma[k] = sopt.brentq(iv_func, 0.0, 10)

        return sigma if n_price > 1 else sigma[0]

    def impvol(self, price, strike, spot, texp, cp_sign=1, setval=False):
        sigma = BsmModel.impvol_base(price, strike, spot, texp, cp_sign, intr=self.intr, divr=self.divr)
        if setval:
            self.sigma = sigma
        return sigma


    def impvol_rational(self, price, strike, spot, texp, cp_sign, setval=False):

        disc_fac = np.exp(-texp * self.intr)
        fwd =  spot  * np.exp(-texp * self.divr) / disc_fac
        beta = BetaFunction(fwd, strike, disc_fac, cp_sign)
        normalised_price = price / np.sqrt(fwd * strike)
        normalised_sigma = beta.impl_vol(normalised_price) 
        sigma = normalised_sigma / np.sqrt(texp)
        if setval:
            self.sigma = sigma

        return sigma

class BsmShiftModel(BsmModel):
    '''
    Basic rule that the definition of the base class is implicitly given,
    so you only redefine(override) what you need to change.
    '''

    shift = None

    def __init__(self, sigma, shift, intr=0.0, divr=0.0, is_fwd=False, disc_fac=None):
        super().__init__(sigma, intr=intr, divr=divr, is_fwd=is_fwd, disc_fac=disc_fac)
        self.shift = shift

    def price(self, strike, spot, texp, cp_sign=1):
        return super().price(strike + self.shift, spot + self.shift, texp, cp_sign=cp_sign)

    def delta(self, strike, spot, texp, cp_sign=1):
        return super().delta(strike + self.shift, spot + self.shift, texp, cp_sign=cp_sign)

    def vega(self, strike, spot, texp, cp_sign=1):
        return super().vega(strike + self.shift, spot + self.shift, texp, cp_sign=cp_sign)

    def gamma(self, strike, spot, texp, cp_sign=1):
        return super().gamma(strike + self.shift, spot + self.shift, texp, cp_sign=cp_sign)

    def impvol(self, price_in, strike, spot, texp, cp_sign=1, setval=False):
        return super().impvol(price_in, strike + self.shift, spot + self.shift, texp, cp_sign=cp_sign)