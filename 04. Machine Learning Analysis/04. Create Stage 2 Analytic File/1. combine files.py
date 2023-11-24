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
pred1 = pred1.select('beneID','state','ageCat','sex','race','houseSize','fedPovLine','speakEnglish','married','UsCitizen','ssi','ssdi','tanf','disabled','enrollMonth','enrollYear','state','enrollMonths','cov2016yes')
print(pred1.count())
print(pred1.printSchema())
pred1.show()

# COMMAND ----------

#predictor 2: stage 1 probability

pred2 = spark.table("dua_058828_spa240.stage2_probability")
pred2 = pred2.select('beneID','state','stage1_lose_coverage_prob')
print(pred2.count())
print(pred2.printSchema())
pred2.show()

# COMMAND ----------

#predictor 3: SDOH features

pred3 = spark.table("dua_058828_spa240.sdoh_features_stage2")
pred3 = pred3.drop('county')
print(pred3.count())
print(pred3.printSchema())
pred3.show()

# COMMAND ----------

#predictor 4: acute care predictors

pred4 = spark.table("dua_058828_spa240.paper1_acute_care_predictors")
pred4 = pred4.drop('x', 'y_allcause', 'y_avoid', 'total_avoid_acute_visits')
print(pred4.count())
print(pred4.printSchema())
pred4.show()

# COMMAND ----------

#predictor 5: pharmacy predictors

pred5 = spark.table("dua_058828_spa240.paper1_pharm_predictors")
pred5 = pred5.drop('x', 'y_pharm')
print(pred5.count())
print(pred5.printSchema())
pred5.show()

# COMMAND ----------

#predictor 6: long-term care

pred6 = spark.table("dua_058828_spa240.paper1_long_term_care_predictors")
#pred6 = pred6.drop('x', 'y_pharm')
print(pred6.count())
print(pred6.printSchema())
pred6.show()

# COMMAND ----------

#predictor 7: dx / ccsr predictors

pred7 = spark.table("dua_058828_spa240.paper1_dx_predictors")
print(pred7.count())
pred7 = pred7.withColumnRenamed("null", "ccsr_null")
print(pred7.printSchema())
pred7.show(1)

# COMMAND ----------

#predictor 8: proc code / betos predictors

pred8 = spark.table("dua_058828_spa240.paper1_proc_code_predictors")
print(pred8.count())
pred8 = pred8.withColumnRenamed("null", "betos_null")
print(pred8.printSchema())
pred8.show(1)

# COMMAND ----------

#predictor 9: rx / ndc agg

pred9 = spark.table("dua_058828_spa240.paper1_rx_predictors")
print(pred9.count())
pred9 = pred9.withColumnRenamed("null", "rx_null")
print(pred9.printSchema())
pred9.show(1)

# COMMAND ----------

#predictor 10: clinician specialty / cms code

pred10 = spark.table("dua_058828_spa240.paper1_clinician_specialty_predictors")
print(pred10.count())
print(pred10.printSchema())
pred10.show(1)

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
for df in [pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, outcome]:
    merged_df = merged_df.join(df, on=["beneID","state"], how='left')

# Show the merged DataFrame
print((merged_df.count(), len(merged_df.columns)))
print(merged_df.printSchema())

# COMMAND ----------

states_to_keep = ["AL", "ME", "MT", "VT", "WY", "IL"]
df = merged_df.filter(col("state").isin(states_to_keep))
df.write.saveAsTable("dua_058828_spa240.stage2_cost_analysis_sample", mode='overwrite')

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = 5000000 / merged_df.count()

# Take a random sample from the DataFrame
sampled_df = merged_df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

# COMMAND ----------

# Show the merged DataFrame
print((sampled_df.count(), len(sampled_df.columns)))
print(sampled_df.printSchema())

# COMMAND ----------

# sampled_df.write.saveAsTable("dua_058828_spa240.stage2_random_sample_5million", mode='overwrite')

# COMMAND ----------

# Assuming pred1 is your PySpark DataFrame containing categorical features
categorical_columns = []  # List to store the categorical feature column names

# Loop through the columns of pred1
for column in pred1.columns[2:]:
    categorical_columns.append(column)

# Print the categorical feature column names
print(categorical_columns)

# COMMAND ----------

# # Assuming pred2 to pred10 are your PySpark DataFrames to be merged
# # Assuming common_columns contains the common column names "beneID" and "state"

# # Initialize the merged DataFrame with pred2
# merged_df = pred1

# # Loop through pred3 to pred10 and perform left join on common columns
# for df in [pred1, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10]:
#     merged_df = merged_df.join(df, on=["beneID","state"], how='left')

# # Show the merged DataFrame
# #merged_df.show()

# # Assuming pred1 is your PySpark DataFrame containing categorical features
# categorical_columns = []  # List to store the categorical feature column names

# # Loop through the columns of pred1
# for column in merged_df.columns[2:]:
#     categorical_columns.append(column)

# # Print the categorical feature column names
# print(categorical_columns)

# COMMAND ----------

# Perform frequency check on the "ageCat" column
freq_check = merged_df.groupBy("ageCat").count().orderBy("count", ascending=False)

# Show the frequency distribution
freq_check.show()

# COMMAND ----------

filtered_df = merged_df.filter(~col("ageCat").isin("under10", "10To17","missing"))
freq_check = filtered_df.groupBy("ageCat").count().orderBy("count", ascending=False)
freq_check.show()

# COMMAND ----------

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = 5000000 / filtered_df.count()

# Take a random sample from the DataFrame
sampled_df = filtered_df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

freq_check = sampled_df.groupBy("ageCat").count().orderBy("count", ascending=False)
freq_check.show()

# COMMAND ----------

sampled_df.write.saveAsTable("dua_058828_spa240.stage2_random_sample_5million_adults", mode='overwrite')

# COMMAND ----------

