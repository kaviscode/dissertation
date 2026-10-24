# Neurons vs Noise: Predicting the FTSE 100 with Neural Networks

A machine learning project investigating whether macroeconomic variables can improve the pricing and prediction of the FTSE 100 Index compared with traditional time-series forecasting.

## Overview

This project was developed as part of my Accounting and Finance dissertation at the University of Leeds. It investigates whether a feedforward neural network can identify nonlinear relationships between UK macroeconomic indicators and FTSE 100 Index prices, and whether incorporating lagged macroeconomic data improves predictive performance.

The study compares a neural network model against alternative specifications using lagged macroeconomic variables and an ARIMA time-series model. The objective is to investigate market predictability and assess whether macroeconomic information provides explanatory power beyond historical index prices alone.

## Research Questions

The dissertation investigates two central questions:

1. **Macroeconomic information:** Can macroeconomic explanatory variables (MEVs) improve the accuracy of FTSE 100 Index price estimates?
2. **Market efficiency:** Does incorporating historical macroeconomic information improve predictive performance, suggesting that markets may not immediately incorporate all available information?

### Data

The study uses monthly UK macroeconomic and financial data spanning January 1984 to December 2024, covering multiple economic and market cycles, including the 2008 Global Financial Crisis, Brexit, and the COVID-19 pandemic.

The dataset combines 26 macroeconomic explanatory variables from sources including the Office for National Statistics, Bank of England, Federal Reserve Economic Data (FRED), Bloomberg, Investing.com, the World Bank, and other economic and financial data providers.

The FTSE 100 Index serves as the dependent variable and proxy for the UK equity market.

### Neural Network Architecture

The primary model, referred to as **BASE**, is a fully connected feedforward neural network built using TensorFlow and Keras.

The architecture uses four hidden dense layers with 512, 256, 128, and 64 units, followed by a single output neuron. It incorporates:

- Leaky ReLU activation functions
- Batch normalization
- Dropout regularization
- L2 regularization
- The Adam optimizer
- Mean squared error (MSE) as the loss function
- Early stopping and adaptive learning-rate reduction

Input and target variables are standardized before training.

### Model Evaluation

The dissertation evaluates model performance using a historical holdout period from July 2020 to December 2024.

The BASE model is compared against:

- **Lagged models:** Neural network specifications using macroeconomic variables from different preceding time intervals.
- **ARIMA:** A traditional autoregressive integrated moving average time-series model used as a benchmark.

Forty BASE model iterations are evaluated to identify the strongest-performing specifications. Performance is assessed using MSE and correlation with actual FTSE 100 Index values.

## Results

The dissertation reports the following headline findings:

| Model / result | Reported performance |
|---|---:|
| BASE model — MSE-based accuracy | 95.28% |
| BASE model — correlation-based accuracy | 88.25% |
| One-month lagged model — MSE-based accuracy | 96.97% |
| ARIMA relative to BASE | 13.05 percentage points lower in MSE-based accuracy and 24.35 percentage points lower in correlation-based accuracy |

The one-month lagged specification outperformed the BASE model on the reported MSE-based measure, suggesting that incorporating recent historical macroeconomic information may improve price estimation.

The neural network models also outperformed the ARIMA benchmark on the reported measures. These findings are consistent with the possibility that macroeconomic variables contain useful information for modelling FTSE 100 Index prices beyond that captured by a conventional univariate time-series approach.

*These are the dissertation's reported results, not independently reproduced results. The accuracy percentages are the study's defined measures and should not be interpreted as guaranteed forecasting accuracy or investment returns.*

## Repository Contents

- `README.md` — project overview, methodology, results, and setup instructions.
- `data.csv` — dataset used by the project.
- `code explanation.xlsx` — line by line explanation of code.
- `model.py` — the model.
- `requirements.txt` — required python libraries.
- `submission.pdf` — submitted paper.

### Prerequisites

- Python
- The packages listed in `requirements.txt`

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

### Running the Model

Run the final model script:

```bash
python <final-model-filename>.py
```

Ensure that `data.csv` is available at the path expected by the script. The exact execution instructions should be updated to match the final implementation committed to this repository.

## Limitations

- **Data frequency and availability:** The analysis is constrained by the availability and consistency of historical macroeconomic data. Some quarterly observations required interpolation to align the datasets.
- **Dataset size:** The study uses a relatively small number of monthly observations for a neural network model with multiple input variables.
- **Model selection:** Selecting the best-performing iteration from repeated experiments on the same holdout period can produce optimistic performance estimates. The reported results should therefore be interpreted cautiously.
- **Generalizability:** Relationships observed in historical UK market data may not persist across different market regimes or future periods.
- **Investment application:** The models are exploratory research tools, not validated trading systems or financial advice.

## Conclusion

This project explores the application of deep learning to financial time-series analysis, combining macroeconomic data with neural network modelling to investigate the predictability of the FTSE 100 Index.

The reported results suggest that macroeconomic variables, particularly when incorporated with a short historical lag, can improve model performance relative to the study's BASE and ARIMA benchmarks. The research also highlights the importance of model validation, data quality, and cautious interpretation when applying machine learning to financial markets.

## Academic Context

This project originated as an undergraduate dissertation for the Accounting and Finance program at the University of Leeds.

**Title:** *Neurons vs Noise: Using Feedforward Neural Networks for Stock Market Index Pricing and Prediction*

