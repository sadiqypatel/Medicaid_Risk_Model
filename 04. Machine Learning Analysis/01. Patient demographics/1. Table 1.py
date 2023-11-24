# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage1_final_analysis")
df = df.withColumn("censusRegion", 
                        when((col("state").isin(['AZ','HI','ID','MT','NM','NV','UT','WA','WY'])), 'West')
                       .when((col("state").isin(['IL','IN','KS','MI','ND'])), 'Midwest')
                       .when((col("state").isin(['AL','KY','LA','MS','TN','WV','VA','MD','DC'])), 'South')                   
                       .otherwise('Northeast')) 

# COMMAND ----------

# Import PySpark and create a SparkSession

spark = SparkSession.builder \
        .appName("ColumnPercentages") \
        .getOrCreate()

# Define a function to calculate column percentages for a single categorical column
def calculate_percentages(df, column_name):
    category_counts = df.groupBy(column_name).agg(count("*").alias("Count"))
    total_rows = df.count()
    category_percentages = category_counts.withColumn("Percentage", round((col("Count") / total_rows) * 100, 1))
    return category_percentages
  
# Read the table into a PySpark DataFrame
#df = spark.table("dua_058828_spa240.stage1_final_analysis")
print((df.count(), len(df.columns)))

# List of categorical columns
categorical_columns = ["ageCat", "sex", "race","censusRegion","houseSize","fedPovLine","speakEnglish","married","UsCitizen","ssi","ssdi","tanf","disabled"]

# Calculate and display column percentages for each categorical column
for column_name in categorical_columns:
    print(f"Column Percentages for {column_name}:")
    calculate_percentages(df, column_name).show()
    
# Stop the Spark session
#spark.stop()

# COMMAND ----------

# Import PySpark and create a SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("mean_months_by_state").getOrCreate()

# Read the table into a PySpark DataFrame
df = spark.table("dua_058828_spa240.finalSample2019")
print((df.count(), len(df.columns)))

# Create a DataFrame from the sample data
total_rows = df.count()

# Group by "state" and calculate the mean of the "month" column
result_df = df.groupBy("state").agg(round(mean("enrolledMonths"), 1).alias("mean_month"))

# Show the result
result_df.show(total_rows, truncate=False)
    
# Stop the Spark session
#spark.stop()

# COMMAND ----------

