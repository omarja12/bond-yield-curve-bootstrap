"""Daily close and volume for five large-cap stocks since 2012 from Yahoo
Finance: cumulative returns, the return correlation matrix, and a round trip
through one CSV file per ticker."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "IBM", "MSFT", "GOOG", "AMZN"]
OUT = Path("data")


def download(tickers=TICKERS, start="2012-01-01"):
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False)
    return pd.concat([raw["Close"], raw["Volume"]], keys=["price", "volume"], axis=1).dropna()


def save_per_ticker(df, tickers=TICKERS):
    OUT.mkdir(exist_ok=True)
    for ticker in tickers:
        df.xs(ticker, axis=1, level=1).to_csv(OUT / f"{ticker}.csv")


def load_prices(tickers=TICKERS):
    frames = [pd.read_csv(OUT / f"{t}.csv", usecols=["Date", "price"], index_col="Date")
              .rename(columns={"price": t}) for t in tickers]
    return pd.concat(frames, axis=1)


if __name__ == "__main__":
    df = download()
    save_per_ticker(df)
    price = load_prices()
    returns = price.pct_change().dropna()

    print(returns.corr().round(3))

    ax = (returns + 1).cumprod().plot(figsize=(10, 5), title="Cumulative returns since 2012")
    ax.set_ylabel("Growth of 1")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_returns.png", dpi=120)
    plt.show()
