# Alcohol Sales Analysis in Iowa (PySpark + ARIMA)

## Objective

This project analyzes historical alcohol sales data from Iowa using **PySpark** for scalable data processing and **ARIMA** for time series forecasting. The main goals are to:

- Analyze trends in alcohol sales (sales statistics (min, max, avg), bottles sold (min, max, avg), sales by city, category, top 10 stores by total sales) 
- Identify the most popular type of alcohol  
- Forecast future sales using ARIMA  

This assignment was completed individually as part of the **Big Data & Analytics** course and demonstrates the application of Big Data tools and predictive analytics.

---

## Problem Description

Alcohol sales data offers valuable insights into consumer behavior and seasonal demand. Using PySpark and time series modeling, this project aims to:

- Efficiently process a large dataset using distributed computing  
- Identify category trends and the most popular alcohol type  
- Forecast future alcohol sales to support planning and decision-making

## Dataset

The dataset is from the [Iowa Liquor Sales database](https://data.iowa.gov/Sales/Iowa-Liquor-Sales/m3tr-qhgy), which includes:

- Sale date  
- Product category  
- Volume sold (liters)  
- Sale amount (USD)  
- Store details and location
- etc.

---

## Key Steps

### 1. Data Processing with PySpark
- Loaded and cleaned raw CSV data using PySpark  
- Converted and parsed date fields for time-based analysis  
- Grouped and aggregated data by month 

### 2. Trend & Popularity Analysis
- Identified sales trends (sales statistics (min, max, avg), bottles sold (min, max, avg)
- Analyzed sales by city, category, etc.
- Determined the **most popular alcohol type** in Iowa

### 3. Forecasting with ARIMA
- Exported monthly sales data from PySpark to Pandas  
- Applied the **ARIMA model** from `statsmodels` to forecast future sales  
- Visualized actual vs predicted values using `matplotlib` 

---

## Repository Structure

1) Task 6.py - the code file for completing 6 assignment.
2) Monthly Sales.png - graph of the monthly sales of alcohol in dollars in Iowa.
3) Monthly Sales Forecast - ARIMA.png - graph of the 12 month forecast of sales of alcohol in dollars in Iowa with an ARIMA model.
4) Whiskey Monthly Forecast - ARIMA.png - graph of the 12 month forecast of sales of canadian whiskies in dollars in Iowa with an ARIMA model.
