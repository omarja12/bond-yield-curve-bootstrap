"""Bootstrap a zero-coupon curve from coupon bond prices.

A ``YieldCurve`` takes an (n x 3) array of [maturity in years, dirty price,
annual coupon rate] for bonds maturing in years 1..n, and derives discount
factors three ways, then spot rates, yields to maturity and 1-year forwards.
"""
import numpy as np
import numpy_financial as npf
from scipy.optimize import minimize

METHODS = ("Matrix operations", "Global Solver", "Iterative Procedure")


class YieldCurve:

    def __init__(self, bonds):
        self.bonds = np.asarray(bonds, dtype=float)
        n = self.bonds.shape[0]
        # Cash-flow matrix: row i holds bond i's flows per 100 face in years 1..n.
        cash_flows = [[np.round(100*row[2], 2) if t < int(row[0]) else np.round((row[2] + 1)*100, 2)
                       for t in range(1, int(row[0]) + 1)] + [0]*(n - int(row[0]))
                      for row in self.bonds]
        self.cash_flows = np.array(cash_flows)
        self.maturities = self.bonds[:, 0]
        self.prices = self.bonds[:, 1]

    def DiscountFactors(self, method="Matrix operations"):
        if method == "Matrix operations":
            # Prices = CF @ DF, an upper-triangular-in-time linear system.
            return np.linalg.solve(self.cash_flows, self.prices)

        if method == "Global Solver":
            def error(dfs):
                return ((self.cash_flows @ dfs - self.prices)**2).sum()
            return minimize(error, np.ones(len(self.prices)), method="BFGS",
                            options={"gtol": 1e-10})["x"]

        if method == "Iterative Procedure":
            # Each bond adds one unknown: solve for it given the earlier ones.
            tmp = []
            for row in self.bonds:
                tmp.append((row[1] - 100*row[2]*sum(tmp)) / (100*(row[2] + 1)))
            return np.array(tmp)

        raise ValueError(f"method must be one of {METHODS}")

    def SpotRates(self, method="Matrix operations"):
        """Annually compounded spot rate for each maturity."""
        dfs = self.DiscountFactors(method)
        return np.power(1./dfs, 1/self.maturities) - 1

    def BondsYtm(self):
        """Yield to maturity of each bond, as the IRR of its cash flows."""
        return np.array([npf.irr([-row[1]] + [row[2]*100 if t < int(row[0]) else (row[2] + 1)*100
                                               for t in range(1, int(row[0]) + 1)])
                         for row in self.bonds])

    def ForwardRates(self):
        """1-year forward rate starting in each of years 1..n-1, annually compounded:
        f(t, t+1) = DF(t) / DF(t+1) - 1."""
        dfs = self.DiscountFactors("Matrix operations")
        return dfs[:-1] / dfs[1:] - 1


# The ten Treasury bonds used throughout: maturities 1-10 years.
MATURITIES = np.arange(1, 11)
PRICES = np.array([96.6, 93.71, 91.56, 90.24, 89.74, 90.04, 91.09, 92.82, 95.19, 98.14])
COUPONS = np.linspace(0.015, 0.0375, num=10)
BONDS = np.column_stack((MATURITIES, PRICES, COUPONS))
