# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import numpy as np

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage1_final_analysis")
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

df.groupBy("enrolled").count().show()

# COMMAND ----------

# Create a new column "loseCoverage" based on the values of the "enrolled" column
df = df.withColumn("loseCoverage", when(df["enrolled"] == 1, 0).otherwise(1))
df.groupBy("loseCoverage").count().show()

# Cast the "label" column to DoubleType
df = df.withColumn("loseCoverage", df["loseCoverage"].cast(DoubleType()))
print(df.count())

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit

# Define a list of month names
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Create a new column "indicator" based on the "first" column
for i, month_name in enumerate(month_names):
    col_num1 = i + 1
    col_num2 = i + 13
    df = df.withColumn(
        "enrollMonth",
        when(df["first"] == f"col{col_num1}", lit(month_name))
        .when(df["first"] == f"col{col_num2}", lit(month_name))
        .otherwise(df["enrollMonth"] if "enrollMonth" in df.columns else None)
    )

# Show the result
df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit

# Define the column numbers and corresponding years
year_mapping = {
    (1, 12): 2017,
    (13, 24): 2018
}

# Create a new column "enrollYear" based on the "first" column
for (start_col, end_col), year in year_mapping.items():
    df = df.withColumn(
        "enrollYear",
        when(df["first"].isin([f"col{i}" for i in range(start_col, end_col + 1)]), lit(year))
        .otherwise(df["enrollYear"] if "enrollYear" in df.columns else None)
    )

# Show the result
df.show()

# COMMAND ----------

from pyspark.sql import SparkSession

# Extract the distinct values of the "beneID" column
distinct_beneID = df.select("beneID","state").distinct()

# Randomly split the distinct "beneID" values into two sets
distinct_beneID1, distinct_beneID2 = distinct_beneID.randomSplit([0.5, 0.5], seed=42)

# Use the split "beneID" sets to filter the original DataFrame and create two separate DataFrames
df1 = df.join(distinct_beneID1, on="beneID", how="inner")
df2 = df.join(distinct_beneID2, on-"beneID", how="inner")

# Check the number of records in each split
print("Number of records in df1:", df1.count())
print("Number of records in df2:", df2.count())

# COMMAND ----------

test  = df1.select("beneID").distinct()
print((test.count(), len(test.columns)))
print((df1.count(), len(df1.columns)))

# COMMAND ----------

test  = df2.select("beneID").distinct()
print((test.count(), len(test.columns)))
print((df2.count(), len(df2.columns)))

# COMMAND ----------

df1.write.saveAsTable("dua_058828_spa240.stage1_sample", mode="overwrite")
df2.write.saveAsTable("dua_058828_spa240.stage2_sample", mode="overwrite")

# COMMAND ----------

df1 = spark.table("dua_058828_spa240.stage1_sample")
df2 = spark.table("dua_058828_spa240.stage2_sample")

# Count unique values in df1
unique_values_df1 = df1.select("state").distinct().count()

# Count unique values in df2
unique_values_df2 = df2.select("state").distinct().count()

# Print the results
print("Unique values in df1: ", unique_values_df1)
print("Unique values in df2: ", unique_values_df2)

# COMMAND ----------

stage1 = spark.table("dua_058828_spa240.stage1_sample")
df = spark.table("dua_058828_spa240.stage2_sample")
print((stage1.count(), len(stage1.columns)))
print((df.count(), len(df.columns)))

# COMMAND ----------

