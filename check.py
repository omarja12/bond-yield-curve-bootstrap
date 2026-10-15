"""Offline checks for the yield-curve maths."""
import numpy as np

from yield_curve import BONDS, METHODS, YieldCurve

y = YieldCurve(BONDS)
dfs = {m: y.DiscountFactors(m) for m in METHODS}

# All three methods agree and reprice every bond.
for m, d in dfs.items():
    assert np.allclose(d, dfs["Matrix operations"], atol=1e-6), m
    assert np.allclose(y.cash_flows @ d, y.prices, atol=1e-4), m

# Discount factors fall with maturity and lie in (0, 1].
d = dfs["Matrix operations"]
assert np.all(np.diff(d) < 0) and np.all((d > 0) & (d <= 1))

# Spot rates reproduce the discount factors.
assert np.allclose((1 + y.SpotRates())**-y.maturities, d)

# Forwards chain the discount factors: DF(n) = DF(1) / prod(1 + f).
assert np.isclose(d[-1], d[0] / np.prod(1 + y.ForwardRates()))

# Each YTM prices its bond.
for (T, price, c), ytm in zip(BONDS, y.BondsYtm()):
    flows = np.full(int(T), 100*c); flows[-1] += 100
    assert np.isclose(np.sum(flows / (1 + ytm)**np.arange(1, int(T) + 1)), price)

# A second curve is solved from its own bonds, not from another instance.
other = BONDS.copy(); other[:, 1] -= 2
z = YieldCurve(other)
assert np.allclose(z.cash_flows @ z.DiscountFactors("Global Solver"), z.prices, atol=1e-4)

print("all checks passed")
