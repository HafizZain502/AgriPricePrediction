# Price Prediction Model for Agricultural Products

## Project Overview

This repository contains the Python notebooks and scripts used to develop a machine learning model for predicting agricultural commodity prices. The work was undertaken as the final‑year thesis for a Master of Science in Business Analytics. The motivation was to provide farmers, traders and policymakers with data‑driven insights into price movements, thereby facilitating better planning, risk management and market stability. In many regions, agricultural prices are subject to volatility driven by seasonal factors, macroeconomic conditions and climate variability.

## Objectives

The primary objectives of the project were:

* Develop a comprehensive price prediction model that integrates economic indicators, historical price data and climate effects into forecasts.
* Evaluate multiple machine learning algorithms – including Random Forest, Support Vector Machines, linear regression, Gradient Boosting, AdaBoost and shallow/deep neural networks – to identify the most effective approach for agricultural price prediction.
* Analyse the influence of socioeconomic and climate variables on commodity prices and capture interdependencies between different agricultural products.

## Dataset

The analysis was based on a dataset comprising daily prices for seventeen vegetables and one fruit (banana). Each commodity was sampled across multiple years, creating a perennial time series. To capture wider market influences, several non‑agricultural variables were included:

* Economic Policy Uncertainty Index – measuring macroeconomic sentiment.
* Petrol prices – reflecting transportation and input costs.
* USD/PKR exchange rate – affecting import costs for fertilizers and machinery.
* Temperature and rainfall – climate variables influencing yields and supply.

Potato prices were selected as the dependent variable in the primary experiments because potatoes are widely consumed and their price fluctuations capture interactions between supply, demand and external factors.

## Methodology

The project followed a structured pipeline:

### 1. Data Cleaning and Feature Engineering

* Data imputation: Missing values in the commodity price columns were filled using time‑series backward filling, preserving the temporal continuity of the dataset.
* Date features: Year, month and seasonal indicators were extracted from the `Date` column.
* Lag features: Previous price values were introduced to capture temporal dependencies.

### 2. Exploratory Data Analysis (EDA)

* Seasonal patterns: Time‑series plots highlighted that potato prices rise from January, peak around July and decline toward year‑end.
* Correlation analysis: The correlation matrix revealed strong positive correlations between certain commodities (e.g., potatoes and eggs at 0.84) and identified the influence of macroeconomic factors like fuel prices and exchange rate on multiple goods.

### 3. Model Selection and Training

Multiple models were implemented using scikit‑learn:

* Random Forest Regressor – an ensemble method selected for its ability to handle non‑linear relationships and high‑dimensional data.
* Support Vector Regressor (SVR) – evaluated with radial basis and linear kernels.
* Linear Regression – used as a baseline and extended with regularization (Lasso).
* Gradient Boosting and AdaBoost Regressors – to compare ensemble techniques.
* Shallow and multi‑layer neural networks (SLNN and MLNN) – implemented via scikit‑learn’s `MLPRegressor`.

Models were trained using an 80/20 train–test split and evaluated with k‑fold cross‑validation to prevent overfitting. Hyperparameter tuning (grid search) was conducted to optimize the Random Forest and SVM models.

### 4. Evaluation Metrics

Performance was assessed using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R‑squared (R²) – indicating the proportion of variance explained by the model.

## Results

The Random Forest Regressor consistently outperformed the other models. On the test set it achieved an R² of 0.962 and MAE of 1.468, with tuning further reducing the error (MAE 1.447 and RMSE 2.701). Gradient Boosting and neural networks delivered respectable performance but did not surpass the tuned Random Forest. Traditional linear regression and SVM models struggled with the non‑linearity of the data.

The analysis confirmed that macro‑economic variables (petrol prices, exchange rates, policy uncertainty) and climate indicators (temperature, rainfall) are important drivers of price movements. Cross‑commodity relationships were also significant; for example, positive correlations between potatoes, onions and eggs suggest common demand patterns.

## Achievements

* Demonstrated that ensemble machine learning techniques can provide accurate and robust predictions of agricultural commodity prices.
* Identified key predictors and revealed the importance of integrating economic and climate variables in price forecasting models.
* Provided actionable insights for farmers, traders and policymakers by highlighting how predictive analytics can support crop planning, trading strategies and policy formulation.

## Usage

To reproduce the results or build upon this work:

1. Clone the repository and install the required Python libraries listed in `requirements.txt` (or install `pandas`, `scikit‑learn`, `matplotlib`, `seaborn` and `numpy`).
2. Obtain the dataset (`Final Data.csv`) and place it in the appropriate directory. The notebook assumes a file path defined in the code (for example `/content/Final Data.csv`).
3. Run the Jupyter notebook (`Price Prediction model.ipynb`) to perform data loading, analysis and model training. The notebook is structured to be executed sequentially.
4. Adjust hyperparameters or add new models as desired. The notebook includes examples of grid search for tuning the Random Forest and SVM models.
5. Review the generated plots and performance tables to understand the model outputs.

## Future Work

The thesis identified several avenues for improvement:

* Enhanced data sources: Incorporating real‑time weather data, satellite imagery and broader socioeconomic indicators could enrich the feature set.
* Advanced time‑series models: Exploring long short‑term memory (LSTM) networks or ARIMA models may better capture temporal dependencies in price data.
* Regional adaptation: The current model is trained on data from Islamabad; applying and validating the approach on datasets from other regions would improve generalizability.
