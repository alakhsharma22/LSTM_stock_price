# Apple Inc. (AAPL) Stock Price Prediction Using LSTM Neural Networks

## Project Overview

This project leverages a Long Short-Term Memory (LSTM) neural network to predict the next-day opening price of Apple Inc. (AAPL) stock. LSTM, a type of recurrent neural network (RNN), effectively captures long-term dependencies and patterns in sequential data, making it highly suitable for time-series forecasting tasks like stock price prediction.

The model is enriched with technical indicators—including Moving Averages (MA), Bollinger Bands, Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD)—to better capture market trends and behaviors.

---

## Detailed Explanation of LSTM Mechanism

Long Short-Term Memory (LSTM) neural networks address the vanishing gradient problem commonly encountered in traditional recurrent neural networks (RNNs). They achieve this through specialized gate mechanisms designed to control the flow of information and retain important information over extended sequences.

At each time step, the LSTM updates its internal states:

* **Input Gate**: Decides what new information from the current input should be added to the network’s memory.
* **Forget Gate**: Controls which existing information in memory should be retained or discarded.
* **Output Gate**: Determines which part of the memory should be outputted.
* **Cell State**: Stores the relevant information learned from previous steps, allowing for long-term memory.
* **Hidden State**: Provides the output used by subsequent layers or time steps.

These gates operate collectively, enabling the LSTM to efficiently learn complex patterns from sequential data such as stock prices.

---

## Technical Indicators Utilized

* **Moving Averages (MA)**:

  * This indicator calculates the average closing price of a stock over a specified period. It helps smooth out short-term fluctuations and highlights longer-term trends.

* **Bollinger Bands**:

  * Bollinger Bands consist of a center line (the moving average) and two bands above and below it (representing standard deviations from the moving average). They indicate the stock’s volatility. When the bands widen, it suggests increased volatility, while narrowing bands indicate decreased volatility.

* **Relative Strength Index (RSI)**:

  * RSI measures the speed and magnitude of recent price changes to evaluate whether a stock is overbought or oversold. It ranges from 0 to 100, with values above 70 typically considered overbought and values below 30 considered oversold.

* **MACD (Moving Average Convergence Divergence)**:

  * MACD is calculated by subtracting a longer-term exponential moving average (EMA) from a shorter-term EMA. It signals potential buying and selling opportunities when it crosses above or below its signal line (another EMA of the MACD line).

---

## Usage Instructions

The script performs the following:

* Fetches historical stock price data (Open, High, Low, Close, Volume) from Yahoo Finance.

* Calculates and adds technical indicators (Moving Averages, Bollinger Bands, RSI, and MACD) to the dataset.

* Trains an LSTM neural network model to predict the next day's Open, High, Low, and Close prices.

* Evaluates the performance of the model using standard metrics.

* Outputs the predictions and visualizes the training loss through plotted curves.
