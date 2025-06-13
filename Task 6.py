import os
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, regexp_replace, sum as _sum, avg, max as _max, min as _min
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA


#Additonal part because I had troubles with pyspark. The only way it worked was when I specified Local IP, used findspark and created an environment
# with python 3.10 (I use 3.12 usually)

#os.environ["SPARK_LOCAL_IP"] = "IP"
#os.environ["PYSPARK_PYTHON"] = r"C:\.....\pyspark_env\python.exe"
#os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\....\pyspark_env\python.exe"

findspark.init()

#Create Spark session
spark = SparkSession.builder.appName("IowaLiquorSales").getOrCreate()


#Data loading and pre-processing

#Read the data with multiLine and escape because just using spark.read showed that there were line symbols inside the rows for one value for each value. 
df = (
 spark.read
 .option("header", True)
 .option("multiLine", True)
 .option("escape", "\"")
 .csv(r'C:\.....\Iowa_Liquor_Sales.csv')
)

#Turned Date column to date by format. Named it "Date"
df = df.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy"))

#Removed $ signs inside the Sale (Dollars) column to keep just the number and cast it from string to number. Named it "Sales"
df = df.withColumn("Sales", regexp_replace(col("Sale (Dollars)"), "[$,]", "").cast("double"))

#Cleaned so no NA values were in Date and Sales
df = df.filter(col("Date").isNotNull() & col("Sales").isNotNull())

#Just to see what type of data I am working with (this was done before the cleaning too)
df.show(10, truncate=False)
df.printSchema()

#Descriptive statistics

#Some sales statistics
df.select(
  _sum("Sales").alias("Total_Sales"), #Summed up the sales to a total. Named it "Total_Sales"
  avg("Sales").alias("Average_Sales"), #Averaged out the sales. Named it "Average_Sales"
  _max("Sales").alias("Max_Sales"), #Max of sales. Named it "Max_Sales"
  _min("Sales").alias("Min_Sales"), #Min of sales. Named it Min_Sales
  _sum("Bottles Sold").alias("Total_Bottles_Sold"), #Summed up bottles sold to a total. Named it "Total_Bottles_Sold"
  avg("Bottles Sold").alias("Average_Bottles_Sold") #Averaged out bottles sold. Named it "Average_Bottles_Sold"
).show()

#Added year and month columns from the date column
df = (
 df.withColumn("Year", year(col("Date")))
   .withColumn("Month", month(col("Date")))
)

#Calculation of monthly sales
monthly_sales = (
 df.groupBy("Year", "Month") #Grouped by year and month
   .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
   .orderBy("Year", "Month") #Order by year and month
)
monthly_sales.show()

#Sales by city in Iowa
sales_by_city = (
 df.groupBy("City") #Grouped by city
    .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
    .orderBy(_sum("Sales").desc()) #Order by sum of sales
)
sales_by_city.show(10)

#Sales by alcohol category name
sales_by_category = (
  df.groupBy("Category Name") #Grouped by category of alcohol name
    .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
    .orderBy(_sum("Sales").desc()) #Order by sum of sales
)
sales_by_category.show(10)

#Top 10 stores by sales
top_stores = (
  df.groupBy("Store Number", "Store Name") #Grouped by store number and name
    .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
    .orderBy(col("Total_Sales").desc()) #Order by sum of sales
    .limit(10) #Limit to 10 stores
)
print("Top 10 stores by sales:")
top_stores.show(truncate=False)

#Top 10 cities by sales
top_cities = (
  df.groupBy("City") #Grouped by city
    .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
    .orderBy(col("Total_Sales").desc()) #Order by sum of sales
    .limit(10) #Limit to 10 stores
)
print("Top 10 cities by sales:")
top_cities.show(truncate=False)

#Top 10 alcohol categories by sales
top_categories = (
  df.groupBy("Category", "Category Name") #Grouped by alcohol category and category name
    .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
    .orderBy(col("Total_Sales").desc()) #Order by sum of sales
    .limit(10) #Limit to 10 stores
)
print("Top 10 alcohol categories by sales:")
top_categories.show(truncate=False)

#Monthly sales

monthly_pd = monthly_sales.toPandas() #Switching to Pandas for easier visualisation
monthly_pd['Date'] = pd.to_datetime(monthly_pd[['Year', 'Month']].assign(DAY=1)) #Date (Year and Month) to Pandas

#Monthly sales visual specifications
plt.figure(figsize=(14, 6))
sns.lineplot(data=monthly_pd, x='Date', y='Total_Sales')
plt.title('Monthly Total Sales Over Time')
plt.ylabel('Total Sales ($)')
plt.xlabel('Date')
plt.show()


#Preparing sales data and date for forecasting 
monthly_pd = monthly_sales.toPandas() #Moving monthly sales to pandas
monthly_pd['Date'] = pd.to_datetime(monthly_pd[['Year', 'Month']].assign(DAY=1)) #Date (Year and Month) to Pandas
monthly_pd = monthly_pd.set_index('Date').sort_index() #Adding index to get a linear timeline

#Checking to see if everything is correct
print(monthly_pd.head())

#ARIMA forecasting monthly sales

#ARIMA(p=4, d=1, q=0) was tried.
#ARIMA(p=3, d=1, q=0) was tried.
#ARIMA(p=2, d=1, q=0) was tried.
#ARIMA(p=1, d=1, q=0) was tried.
#Fitted ARIMA(p=5, d=1, q=0) to the Total_Sales column. This ARIMA model had the best results.
model = ARIMA(monthly_pd['Total_Sales'], order=(5, 1, 0))
model_fit = model.fit()

#Printed model summary
print(model_fit.summary())

#12 month forecast

#Forecasted the next 12 time steps (1 step = 1 month because data is in months now)
forecast = model_fit.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean #Forecast by mean
conf_int = forecast.conf_int()

#Date index was created to have linear time
forecast_index = pd.date_range(
  start=monthly_pd.index[-1] + pd.offsets.MonthBegin(1),
  periods=12,
  freq='MS' #MS - start of month
)

#Forecast values were aligned with the date index
forecast_series = pd.Series(forecast_mean.values, index=forecast_index)
conf_int.index = forecast_index

#Forecast visual specifications
plt.figure(figsize=(12, 6))
plt.plot(monthly_pd['Total_Sales'], label='Historical Sales')
plt.plot(forecast_series, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title('Monthly Sales Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#ARIMA forecasting canadian whiskies monthly sales

#Only kept the data with category name = canadian whiskies
whisky_df = df.filter(col("Category Name") == "CANADIAN WHISKIES")

#Added year and month columns from the date column
whisky_df = (
  whisky_df.withColumn("Year", year(col("Date")))
           .withColumn("Month", month(col("Date")))
)

#Created monthly sales for canadian whiskies
monthly_whisky_sales = (
  whisky_df.groupBy("Year", "Month") #Grouped by year and month
           .agg(_sum("Sales").alias("Total_Sales")) #Aggregate sum of sales
           .orderBy("Year", "Month") #Order by year and month
)

#Preparing sales data and date for forecasting 
monthly_whisky_pd = monthly_whisky_sales.toPandas() #Moving monthly sales to pandas
monthly_whisky_pd['Date'] = pd.to_datetime(monthly_whisky_pd[['Year', 'Month']].assign(DAY=1)) #Date (Year and Month) to Pandas
monthly_whisky_pd = monthly_whisky_pd.set_index('Date').sort_index() #Adding index to get a linear timeline

#ARIMA(p=4, d=1, q=0) was tried.
#ARIMA(p=3, d=1, q=0) was tried.
#ARIMA(p=2, d=1, q=0) was tried.
#ARIMA(p=1, d=1, q=0) was tried.
#Fitted ARIMA(p=5, d=1, q=0) to the Total_Sales column. This ARIMA model had the best results.
model = ARIMA(monthly_whisky_pd['Total_Sales'], order=(5, 1, 0))
model_fit = model.fit()

#Printed model summary
print(model_fit.summary())

#Forecasted the next 12 time steps (1 step = 1 month because data is in months now)
forecast = model_fit.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean #Forecast by mean
conf_int = forecast.conf_int()

#Date index was created to have linear time
forecast_index = pd.date_range(
  start=monthly_whisky_pd.index[-1] + pd.offsets.MonthBegin(1),
  periods=12,
  freq='MS' #MS - start of month
)

#Forecast values were aligned with the date index
forecast_series = pd.Series(forecast_mean.values, index=forecast_index)
conf_int.index = forecast_index

#Forecast visual specifications
plt.figure(figsize=(12, 6))
plt.plot(monthly_whisky_pd['Total_Sales'], label='Historical Sales')
plt.plot(forecast_series, label='Forecast', color='red')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label='95% Confidence Interval')

plt.title('Monthly Sales Forecast for Canadian Whiskies (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Stopping the Spark session
spark.stop()