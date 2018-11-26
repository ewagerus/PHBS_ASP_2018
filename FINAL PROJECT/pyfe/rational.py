import math
import numpy as np
from scipy import stats as ss
from scipy import misc as sm

'''The code below was written based on the article "Let's be Rational" by Peter Jackel (2016). 
Each section of the code contains a reference number which will allow the reader to navigate 
between the article and code and easily identify the relevant formulas.'''

def is_below_horizon(x):

    return np.abs(x) < 1e-6

class BetaFunction(object):

    def __init__(self, forward, strike, disc_fac,cp_sign):
        
        self.forward = forward
        self.strike = strike
        self.disc_fac = disc_fac
        self.cp_sign = cp_sign
        self.n_x = np.log(forward / strike)
        self.b_max = np.exp(self.cp_sign * self.n_x / 2)
        self._get_lower_center_upper_point()

        self.f_l = self._func_lower_map()
        self.f_u = self._func_upper_map()
        self.f_cr = self._func_cr_map()
        self.f_cl = self._func_cl_map()
                               
    def sigma(self, price):
        if price < 0 or price > self.b_max:
            raise ValueError(
                'Price is out of the range, price should be larger than 0 and smaller than {}'.format(self.b_max))
        if price < self.b_l:
            return self.f_l(price)
        elif price < self.b_c:
            return self.f_cl(price)
        elif price < self.b_u:
            return self.f_cr(price)
        elif price < self.b_max:
            return self.f_u(price)

    def normalised_bsm(self, sigma=None):

        #2.4
        cp_sign = self.cp_sign
        x = self.n_x
        return cp_sign * self.disc_fac * (np.exp(x/2)*ss.norm.cdf(cp_sign*(x/sigma + sigma/2))\
         - np.exp(-0.5*x)*ss.norm.cdf(cp_sign *(x/sigma - sigma/2)))
        


    def normalised_vega(self, sigma=None, x=None):
        #4.6
        x = x if x else self.n_x
        return self.disc_fac * 1 / np.sqrt(2*np.pi) * np.exp(-0.5*((x/sigma)**2 + (sigma/2)**2))

    def _get_lower_center_upper_point(self):

        self.sigma_c = np.sqrt(2 * np.abs(self.n_x))
        self.b_c = self.normalised_bsm(self.sigma_c)

        self.vega_c = self.normalised_vega(self.sigma_c)

        self.sigma_l = self.sigma_c - self.b_c / self.vega_c
        self.sigma_u = self.sigma_c + (self.b_max - self.b_c) / self.vega_c

        self.b_l = self.normalised_bsm(self.sigma_l)
        self.b_u = self.normalised_bsm(self.sigma_u)

        self.vega_l = self.normalised_vega(self.sigma_l)
        self.vega_u = self.normalised_vega(self.sigma_u)

    def _rational_cubic_interpolation(self, x, x_l, x_r, y_l, y_r, d_l, d_r, r):

        h = x_r - x_l

        if is_below_horizon(h):

            return 0.5 * (y_l + y_r)

        else:
            t = (x - x_l) / h
            omt = 1 - t

            return (y_r*t**3 + (r*y_r - h*d_r) * t**2 * omt + (r*y_l + h*d_l)*t * omt**2 + y_l * omt**3) / (1 + (r-3) * t * omt)

    def _control_R(self, x_l, x_r, y_l, y_r, d_l, d_r, dd_l, left=True):
        h = x_r - x_l
        delta = (y_r - y_l) / h
        if left:

            return (0.5 * h * dd_l + (d_r - d_l)) / (delta - d_l)
        else:

            return (0.5 * h * dd_l + (d_r - d_l)) / (d_r - delta)

    def _get_f_map_and_first_two_derivatives(self, sigma, left=True):

        x = self.n_x

        if left:
            z = -np.abs(x) / sigma / np.sqrt(3)
            PHI = ss.norm.cdf(-z)
            phi = ss.norm.pdf(z)

            fpp = np.pi / 6 * z**2 / sigma ** 3 * PHI * np.sqrt(2*z**2 + 0.25 * sigma**2) *\
                (8 * np.sqrt(3) * sigma * np.abs(x) +
                 (3*sigma**3*(sigma**2 - 8) - 8*x**2)*PHI/phi)

            if is_below_horizon(sigma):

                fp = 1
                f = 0
            else:
                fp = 2*np.pi * z**2 * PHI**2 * np.exp(z**2 + sigma**2 / 8)
                if is_below_horizon(x):
                    f = 0
                else:
                    f = 2*np.pi * np.abs(x) * PHI ** 3 / (3 * np.sqrt(3))
            return f, fp, fpp
        else:

            f = ss.norm.cdf(-0.5 * sigma)

            if is_below_horizon(x):
                fp = -0.5
                fpp = 0
            else:
                w = (x/sigma)**2
                fp = -0.5 * np.exp(0.5 * w)
                fpp = np.sqrt(np.pi/2) * np.exp(w +
                                                0.125 * sigma**2) * w / sigma

            return f, fp, fpp

    def _func_cl_map(self):

        r_lc = self._control_R(self.b_l, self.b_c, self.sigma_l,
                               self.sigma_c, 1/self.vega_l, 1/self.vega_c, 0, left=True)

        return lambda price: self._rational_cubic_interpolation(price,
                                                                self.b_l, self.b_c, self.sigma_l, self.sigma_c, 1/self.vega_l, 1/self.vega_c, r_lc)

    def _func_cr_map(self):

        r_rc = self._control_R(self.b_c, self.b_u, self.sigma_c,
                               self.sigma_u, 1/self.vega_c, 1/self.vega_u, 0, left=False)

        return lambda price: self._rational_cubic_interpolation(price,
                                                                self.b_c, self.b_u, self.sigma_c, self.sigma_u, 1/self.vega_c, 1/self.vega_u, r_rc)

    def _func_upper_map(self):

        f_u, d_u, dd_u = self._get_f_map_and_first_two_derivatives(
            self.sigma_u, left=False)

        r_umax = self._control_R(self.b_u, self.b_max,
                                 f_u, 0, d_u, -0.5, dd_u, left=False)

        return lambda price: -2 * ss.norm.ppf(self._rational_cubic_interpolation(price, self.b_u, self.b_max, f_u, 0, d_u, -0.5, r_umax))

    def _func_lower_map(self):

        f_l, d_l, dd_l = self._get_f_map_and_first_two_derivatives(
            self.sigma_l, left=True)
        r_0l = self._control_R(0, self.b_l, 0, f_l, 1, d_l, dd_l, left=True)

        def sigma_func(price):

            if is_below_horizon(price):
                return 0
            else:

                f = self._rational_cubic_interpolation(
                    price, 0, self.b_l, 0, f_l, 1, d_l, r_0l)
                temp = np.sqrt(3) * np.power(0.5 * f /
                                             np.pi / np.abs(self.n_x), 1/3)
                return np.abs(self.n_x / np.sqrt(3) / ss.norm.ppf(temp))

        return sigma_func
    
    def g_sigma(self, price):
        
        #5.2
        b_u_update = np.maximum(self.b_u,(self.b_max/2))
              
        #5.1
        if price < 0 or price > self.b_max:
            raise ValueError(
                'Price is out of the range, price should be larger than 0 and smaller than {}'.format(self.b_max))
        if price < self.b_l:
            return lambda sigma: 1/np.log(self.normalised_bsm(sigma)) - 1/np.log(price)
        elif price < b_u_update:
            return lambda sigma: self.normalised_bsm(sigma) - price
        elif price < self.b_max:
            return lambda sigma: np.log((self.b_max-price)/(self.b_max-self.normalised_bsm(sigma)))

    def sigma_udpate(self, g, sigma):
        dg = sm.derivative(g, x0=sigma, n=1)
        ddg = sm.derivative(g, x0=sigma, n=2) 
        dddg = sm.derivative(g, x0=sigma, n=3, order=5)
        
        #5.7
        vn = -g(sigma)/dg
        gn = ddg/dg
        dn=dddg/dg
        #5.6
        return sigma + vn*((1+0.5*gn*vn)/(1+vn*(gn+(1/6)*dn*vn)))

    def impl_vol(self,price):
        
        g = self.g_sigma(price)
        sigma_0 = self.sigma(price)

        #iteration
        for _ in range(2):
    
            sigma_0 = self.sigma_udpate(g, sigma_0)

        return sigma_0 
        

if __name__ == '__main__':
    
    a = BetaFunction(80, 84, 1)
    price = 0.005
    print(a.sigma(price))
    
    print(a.normalised_bsm(a.sigma(price)))
    g = a.g_sigma(price)
    print(g(a.sigma(price)))
    # print(a.normalised_bsm(a.iml_vol(price)[0], a.n_x))
    print(a.iml_vol(price))
    # print(a.g_sigma(price)(a.sigma(price)))
    print(a.g_sigma(price)(a.iml_vol(price)))
        
    