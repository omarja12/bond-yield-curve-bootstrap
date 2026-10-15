"""Print the bootstrapped curve and save a figure of spot, YTM and forward rates."""
import matplotlib.pyplot as plt
import numpy as np

from yield_curve import BONDS, METHODS, YieldCurve

np.set_printoptions(suppress=True, formatter={"float_kind": "{:0.4f}".format})

y = YieldCurve(BONDS)
for method in METHODS:
    print(f"{method:>20}: {y.DiscountFactors(method)}")

spot, ytm, fwd = y.SpotRates(), y.BondsYtm(), y.ForwardRates()
print(f"{'spot rates':>20}: {spot}")
print(f"{'yields to maturity':>20}: {ytm}")
print(f"{'1y forwards':>20}: {fwd}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y.maturities, 100*spot, color="black", marker="o", label="Spot rate")
ax.plot(y.maturities, 100*ytm, color="green", marker="o", label="Yield to maturity")
# The forward from year t to t+1 is plotted at its end date, t+1.
ax.plot(y.maturities[1:], 100*fwd, color="red", marker="o", label="1y forward rate")
ax.grid(alpha=0.3)
ax.set_xlabel("Maturity in years")
ax.set_ylabel("Rate in %")
ax.set_title("Bootstrapped term structure")
ax.legend()
fig.tight_layout()
fig.savefig("term_structure.png", dpi=120)
plt.show()
