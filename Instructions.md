# Machine learning and AI for intraday return prediction

## 1. Data download

* Download the already cleaned and merged trades and quotes data (from the Daily TAQ database)
    * Company: Citigroup, ticker "C"
    * Time period: Jan 1, 2024 to Jan 31, 2024
* Import the Parquet file into a Pandas dataframe

## 2. Defining variables

* Calculate the following variables for each minute:
    * average relative quoted spread
    * order imbalance
    * average depth imbalance, defined as

      $$\text{Depth imbalance} = \frac{\text{ASKSIZ} - \text{BIDSIZ}}{\text{ASKSIZ} + \text{BIDSIZ}}$$

    * traded volume
    * one minute return from closing mid prices for each minute
    * realized volatility as the sum of squared one minute returns over the past hour
* Winsorize all variables at 1% and 99%

## 3. Random Forest with walk forward testing

* Use the same walk forward strategy, but replace OLS with **Random Forest** estimations (refitted daily):
    * Train a model on a two week window
    * Predict one minute returns for the next day
    * Slide the window one day forward and repeat
    * Hint: think carefully about the parameters `n_estimators` and `min_samples_leaf`
* Store
    * the feature importances from each refit
    * the predicted and actual returns for each minute
* Recalculate $R^2$, RMSE, MAE and correlation, and compare them to the corresponding OLS metrics
* Calculate the average feature importance over all refits and compare it to the corresponding OLS estimations

## 4. Backtesting: Random Forest

* Backtest the RF walk forward strategy (including transaction costs):
    * compute the total cumulative return and plot it on a graph
    * average daily return, standard deviation of daily returns, daily Sharpe ratio
    * hit rate, maximum drawdown, turnover
    * t statistics of the strategy daily returns
* **Question: Compare the strategy performance between RF and OLS (with transaction costs). Would you trade based on your strategy? Why or why not?**

## 5. XGBoost with walk forward testing

* Use the same walk forward strategy, but replace Random Forest with **XGBoost** estimations (refitted daily)
    * Hint: think carefully about the following parameters: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`
* Recalculate $R^2$, RMSE, MAE and correlation, and compare them to the corresponding OLS and RF metrics
* Calculate the average feature importance over all refits and compare it to the corresponding OLS and RF estimations

## 6. Backtesting: XGBoost

* Backtest the XGBoost walk forward strategy (including transaction costs):
    * compute the total cumulative return and plot it on a graph
    * average daily return, standard deviation of daily returns, daily Sharpe ratio
    * hit rate, maximum drawdown, turnover
    * t statistics of the strategy daily returns
* **Question: Compare the strategy performance between XGBoost, RF and OLS (with transaction costs). Which model would you choose: OLS, RF or XGBoost? Why?**

## 7. Evaluating news sentiment with FinBERT

* Scrape Google News RSS for Citigroup and filter strictly for January 2024:
    * Example of a news search for climate change: <https://news.google.com/rss/search?q=climate+change>
    * Make sure to search across several keywords, for example "Citigroup", "Citi", "Citibank", "C", "C stock"
    * Parse the RSS feed (extract "title", "timestamp" and "link")
    * Save the results in a dataframe called `news_items`
* Use the FinBERT sentiment analysis model:

  ```python
  import torch
  from transformers import pipeline
  sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
  ```

* Apply the FinBERT sentiment model to the title column and store the **sentiment label**
* Map the sentiment label to a numeric sentiment signal:
    * "positive" = 1, "neutral" = 0, "negative" = $-1$
* Compute the average sentiment signal for each day

## 8. Random Forest with the "Sentiment signal"

* Merge the daily news sentiment with the existing TAQ data
* Estimate the Random Forest model with `Sentiment_signal` included as an additional feature
* Backtest the Random Forest model with `Sentiment_signal` (with transaction costs)
* **Question: Compare the strategy performance between RF with and without the sentiment signal. Which model would you choose? Why?**
