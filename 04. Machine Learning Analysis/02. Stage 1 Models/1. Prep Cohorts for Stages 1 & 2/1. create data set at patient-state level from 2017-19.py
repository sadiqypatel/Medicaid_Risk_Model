# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import numpy as np

# COMMAND ----------

df2017 = spark.table("dua_058828_spa240.finalSample2017")
df2018 = spark.table("dua_058828_spa240.finalSample2018")
df2019 = spark.table("dua_058828_spa240.finalSample2019")

dfAll = df2017.union(df2018).union(df2019)
print(dfAll.printSchema())
dfAll = df2017.union(df2018).union(df2019)
dfAll = dfAll.select("beneID", "state").distinct()
print(dfAll.printSchema())
print((dfAll.count(), len(dfAll.columns)))

# COMMAND ----------

dfpartial2017 = df2017.select("beneID","state","janElig","febElig","marElig","aprElig","mayElig","junElig","julElig","augElig","sepElig","octElig","novElig","decElig")

dfpartial2017 = dfpartial2017.withColumnRenamed("janElig", "col1").withColumnRenamed("febElig", "col2").withColumnRenamed("marElig", "col3").withColumnRenamed("aprElig", "col4").withColumnRenamed("mayElig", "col5").withColumnRenamed("junElig", "col6").withColumnRenamed("julElig", "col7").withColumnRenamed("augElig", "col8").withColumnRenamed("sepElig", "col9").withColumnRenamed("octElig", "col10").withColumnRenamed("novElig", "col11").withColumnRenamed("decElig", "col12")

print(dfpartial2017.printSchema())
dfpartial2017.show()

# COMMAND ----------

dfpartial2018 = df2018.select("beneID","state","janElig","febElig","marElig","aprElig","mayElig","junElig","julElig","augElig","sepElig","octElig","novElig","decElig")

dfpartial2018 = dfpartial2018.withColumnRenamed("janElig", "col13").withColumnRenamed("febElig", "col14").withColumnRenamed("marElig", "col15").withColumnRenamed("aprElig", "col16").withColumnRenamed("mayElig", "col17").withColumnRenamed("junElig", "col18").withColumnRenamed("julElig", "col19").withColumnRenamed("augElig", "col20").withColumnRenamed("sepElig", "col21").withColumnRenamed("octElig", "col22").withColumnRenamed("novElig", "col23").withColumnRenamed("decElig", "col24")

print(dfpartial2018.printSchema())
dfpartial2018.show()

# COMMAND ----------

dfpartial2019 = df2019.select("beneID","state","janElig","febElig","marElig","aprElig","mayElig","junElig","julElig","augElig","sepElig","octElig","novElig","decElig")

dfpartial2019 = dfpartial2019.withColumnRenamed("janElig", "col25").withColumnRenamed("febElig", "col26").withColumnRenamed("marElig", "col27").withColumnRenamed("aprElig", "col28").withColumnRenamed("mayElig", "col29").withColumnRenamed("junElig", "col30").withColumnRenamed("julElig", "col31").withColumnRenamed("augElig", "col32").withColumnRenamed("sepElig", "col33").withColumnRenamed("octElig", "col34").withColumnRenamed("novElig", "col35").withColumnRenamed("decElig", "col36")

print(dfpartial2019.printSchema())
dfpartial2019.show()

# COMMAND ----------

merged = dfAll.join(dfpartial2017, on=["beneID","state"], how="left")
merged = merged.join(dfpartial2018, on=["beneID","state"], how="left")
merged = merged.join(dfpartial2019, on=["beneID","state"], how="left")
merged.show()
merged = merged.fillna(0)
merged.show()
print(merged.printSchema())

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

# Create a Spark session
spark = SparkSession.builder.appName("split_dataframe").getOrCreate()

# Add a unique ID column to the DataFrame
df = merged.withColumn("ID", monotonically_increasing_id())

# Define the number of splits
num_splits = 100

# Define the weights for the randomSplit method
# In this case, we use equal weights for all splits, but you can adjust the weights as needed
weights = [1.0] * num_splits

# Split the DataFrame into 30 smaller DataFrames
splits = df.randomSplit(weights, seed=42)

# Show the results of the split
for i, split_df in enumerate(splits):
    print(f"Split {i + 1}:")
    #print((split_df.count(), len(split_df.columns)))
    exec(f"split_{i+1} = split_df")
    #split_df.show()

# COMMAND ----------

print((split_1.count(), len(split_1.columns)))
print((split_2.count(), len(split_2.columns)))
print((split_3.count(), len(split_3.columns)))

# COMMAND ----------

split_1.write.saveAsTable("dua_058828_spa240.split_1", mode="overwrite")
split_2.write.saveAsTable("dua_058828_spa240.split_2", mode="overwrite")
split_3.write.saveAsTable("dua_058828_spa240.split_3", mode="overwrite")

# COMMAND ----------

#spark = SparkSession.builder.appName("ReadTable").getOrCreate()
split_1 = spark.table("dua_058828_spa240.split_1")
split_2 = spark.table("dua_058828_spa240.split_2")
split_3 = spark.table("dua_058828_spa240.split_3")
#split_1.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, coalesce, lit

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("LoopThroughColumns").getOrCreate()

# Define a function to process each DataFrame
def process_dataframe(df):
    # Create an expression to find the first column with a value of 1
    first_one_expr = "CASE "
    for i in range(1, 37):  # Include all columns (col1 to col36)
        first_one_expr += f"WHEN col{i} = 1 THEN 'col{i}' "
    first_one_expr += "END"
    
    # Add a column with the name of the first column containing a 1
    df = df.withColumn("first", expr(first_one_expr))
    
    # Calculate the sum of the next 12 columns after the first column with a 1 (excluding the first column)
    sum_expr = "CASE "
    for i in range(1, 25):  # Limit to the first 24 columns (col1 to col24)
        # Determine the number of columns to sum (max of 12 columns)
        num_cols_to_sum = min(12, 37 - (i + 1))
        # Create an expression to sum the next 12 columns (excluding the first column) if the current column is 1
        if num_cols_to_sum > 0:
            sum_expr += f"WHEN col{i} = 1 THEN (" + " + ".join([f"col{j}" for j in range(i + 1, i + 1 + num_cols_to_sum)]) + ") "
    sum_expr += "END"
    
    # Add a column with the sum of the next 12 columns after the first column with a 1 (excluding the first column)
    df = df.withColumn("enrollMonths", coalesce(expr(sum_expr), lit(0)))
    
    return df

# Loop through the DataFrames named split_1 to split_100
for i in range(1, 101):
    # Dynamically access the DataFrame by name using eval
    df = eval(f"split_{i}")
    
    # Process the DataFrame
    df = process_dataframe(df)
    
    # Update the DataFrame variable with the new DataFrame that includes the extra columns
    globals()[f"split_{i}"] = df

# COMMAND ----------

split_1.show()

# COMMAND ----------

split_2.show()

# COMMAND ----------

split_3.show()

# COMMAND ----------

# Initialize an empty DataFrame to store the result
result_df = None

# Loop through the DataFrames named split_1 to split_100
for i in range(1, 101):
    # Dynamically access the DataFrame by name using eval
    df = eval(f"split_{i}")
    
    # If result_df is empty, set it to the current DataFrame
    if result_df is None:
        result_df = df
    # Otherwise, concatenate the current DataFrame to result_df using union
    else:
        result_df = result_df.union(df)

# The DataFrame result_df now contains the concatenated data from all 100 DataFrames

# If you want to verify the result, you can use the show method
# result_df.show()


# COMMAND ----------

print((result_df.count(), len(result_df.columns)))
print((dfAll.count(), len(dfAll.columns)))
#result_df.show()

# COMMAND ----------

result_df.write.saveAsTable("dua_058828_spa240.enrollment_analysis", mode="overwrite")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("DistributionOfEnrollMonths").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrollMonths" column
# If you have multiple DataFrames (split_1 to split_100), you can concatenate them using the "unionByName" function

# Calculate the total number of rows in the DataFrame
total_rows = result_df.count()

# Calculate the distribution of the "enrollMonths" variable
distribution_df = result_df.groupBy("enrollMonths") \
                    .agg(count("*").alias("count")) \
                    .withColumn("percentage", round((col("count") / total_rows) * 100, 2)) \
                    .orderBy("enrollMonths")  # Order the results by "enrollMonths" in ascending order

# Show the distribution and percentage of total rows for each month value
distribution_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("CountValuesInColumns").getOrCreate()

# Assuming "result_df" is the DataFrame containing columns "col1" through "col24"

# Define a function to create a filter expression for a range of columns
def create_filter_expr(start_col, end_col):
    filter_expr = None
    for i in range(start_col, end_col + 1):
        current_expr = col(f"col{i}") == 1
        filter_expr = current_expr if filter_expr is None else filter_expr | current_expr
    return filter_expr

# Count the total number of rows that have a value of 1 in columns col1 through col12
count_col1_to_col12 = result_df.filter(create_filter_expr(1, 12)).count()

# Count the total number of rows that have a value of 1 in columns col13 through col24
count_col13_to_col24 = result_df.filter(create_filter_expr(13, 24)).count()

# Print the counts
print(f"Count of rows with a value of 1 in columns col1 through col12: {count_col1_to_col12}")
print(f"Count of rows with a value of 1 in columns col13 through col24: {count_col13_to_col24}")