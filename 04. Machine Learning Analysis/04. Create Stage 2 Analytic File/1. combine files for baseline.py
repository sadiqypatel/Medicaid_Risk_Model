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

#predictor 1: patient characteristics

pred1 = spark.table("dua_058828_spa240.demographic_stage2")
pred1 = pred1.select('beneID','state','ageCat','sex')
print(pred1.count())
print(pred1.printSchema())
pred1.show()

# COMMAND ----------

#predictor 7: dx / ccsr predictors

pred2 = spark.table("dua_058828_spa240.paper1_dx_predictors")
print(pred2.count())
pred2 = pred2.withColumnRenamed("null", "ccsr_null")
print(pred2.printSchema())
pred2.show(1)

# COMMAND ----------

#predictor 9: rx / ndc agg

pred3 = spark.table("dua_058828_spa240.paper1_rx_predictors")
print(pred3.count())
pred3 = pred3.withColumnRenamed("null", "rx_null")
print(pred3.printSchema())
pred3.show(1)

# COMMAND ----------

#outcome 1: all-caute and non-emergent acute care

outcome = spark.table("dua_058828_spa240.paper1_acute_care_outcome")
print(outcome.count())
print(outcome.printSchema())
outcome.show(1)

# COMMAND ----------

# Assuming common_columns contains the common column names "beneID" and "state"

# Initialize the merged DataFrame with pred2
merged_df = pred1

# Loop through pred3 to pred10 and perform left join on common columns
for df in [pred2, pred3, outcome]:
    merged_df = merged_df.join(df, on=["beneID","state"], how='left')

# Show the merged DataFrame
print((merged_df.count(), len(merged_df.columns)))
print(merged_df.printSchema())

# COMMAND ----------

# Show the merged DataFrame
print((sampled_df.count(), len(sampled_df.columns)))
print(sampled_df.printSchema())

# COMMAND ----------

# Assuming pred1 is your PySpark DataFrame containing categorical features
categorical_columns = []  # List to store the categorical feature column names

# Loop through the columns of pred1
for column in sampled_df.columns[2:]:
    categorical_columns.append(column)

# Print the categorical feature column names
print(categorical_columns)

# COMMAND ----------

