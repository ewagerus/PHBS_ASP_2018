import abc


class OptionModelABC(abc.ABC):
    sigma = None

    @abc.abstractmethod
    def price(self, strike, spot, texp, cp_sign=1):
        return 0.0

    def delta(self, strike, spot, texp, cp_sign=1):
        return 0.0

    def vega(self, strike, spot, texp, cp_sign=1):
        return 0.0

    def gamma(self, strike, spot, texp, cp_sign=1):
        return 0.0

    def impvol(self, price, strike, spot, texp, cp_sign=1):
        return 0.0

    def delta_shock(self, strike, spot, texp, cp_sign=1):
        return spot*0.001  # 10 bps of spot price

    def gamma_shock(self, strike, spot, texp, cp_sign=1):
        return spot*0.001  # 10 bps of spot price

    def delta_numeric(self, strike, spot, texp, cp_sign=1):
        h = self.delta_shock(strike, spot, texp, cp_sign)
        delta = (self.price(strike, spot+h, texp, cp_sign)-self.price(strike, spot-h, texp, cp_sign))/(2*h)
        return delta

    def gamma_numeric(self, strike, spot, texp, cp_sign=1):
        h = self.gamma_shock(strike, spot, texp, cp_sign)
        gamma = (self.price(strike, spot+h, texp, cp_sign) - 2*self.price(strike, spot, texp, cp_sign)
                 + self.price(strike, spot-h, texp, cp_sign))/(h*h)
        return gamma

    def price_from_contract(self, spot, contract):
        return self.price(contract.strike, spot, contract.texp(), contract.cp_sign)