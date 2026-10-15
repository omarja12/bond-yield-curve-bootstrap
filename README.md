# Bond Yield Curve Bootstrap

Bootstraps a zero-coupon term structure from ten coupon-bearing Treasury bonds (maturities 1 to 10 years), then derives spot rates, yields to maturity and 1-year forward rates. A second script pulls daily prices for five large-cap stocks from Yahoo Finance and looks at cumulative returns and correlations.

## Yield curve

`YieldCurve` takes an `n x 3` array of `[maturity, dirty price, coupon rate]` and builds the cash-flow matrix `CF` (one row per bond, one column per year), so prices satisfy `P = CF · DF`.

| Method | How the discount factors are found |
|---|---|
| `Matrix operations` | solve the linear system `CF · DF = P` directly |
| `Global Solver` | minimise squared pricing error with BFGS |
| `Iterative Procedure` | strip one maturity at a time, each bond adding one unknown |

From the discount factors:

- spot rate (annual compounding): `s(T) = DF(T)^(-1/T) - 1`
- yield to maturity: the IRR of each bond's cash flows
- 1-year forward from `t` to `t+1`: `f = DF(t) / DF(t+1) - 1`

```bash
pip install -r requirements.txt
python plot_rates.py    # prints all three sets of discount factors, saves term_structure.png
python check.py         # offline checks
```

`check.py` asserts that the three methods agree and reprice every bond, that spot rates and forwards reproduce the discount factors, that each yield prices its bond, and that every curve is solved from its own bonds.

## Equities

```bash
python equities.py
```

Downloads close price and volume for AAPL, IBM, MSFT, GOOG and AMZN since 2012, writes one CSV per ticker to `data/`, reads them back into a single price table, prints the correlation matrix of daily returns and saves `cumulative_returns.png`. It needs network access to Yahoo Finance.

## CI

The GitHub Actions workflow runs the checks and both scripts on every push and uploads the figures. The equities step is allowed to fail, since it depends on Yahoo Finance being reachable.
