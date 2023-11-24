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

test_df = spark.table("dua_058828_spa240.test_xg_boost_stage2_nonemerg2")
df = spark.table("dua_058828_spa240.stage2_random_sample_5million")

print((test_df.count(), len(test_df.columns)))
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

df = df.select("beneID","state","race")

# Count the occurrences of each race
race_counts = df.groupBy("race").agg(count("*").alias("count"))

# Show the results
race_counts.show()

# COMMAND ----------

# Create the new "race_category" column based on the conditions
df = df.withColumn(
    "race_category",
    when(col("race") == "white", "White")
    .when(col("race") == "missing", "missing")
    .when(col("race") == "black", "Black")
    .when(col("race") == "hispanic", "Hispanic")
    .otherwise("minority")
)

# Count the occurrences of each race
race_counts = df.groupBy("race_category").agg(count("*").alias("count"))

# Show the results
race_counts.show()

# COMMAND ----------

# Perform the left merge on 'beneID' and 'state' columns
print((test_df.count(), len(test_df.columns)))
df = test_df.join(df, on=['beneID', 'state'], how='left')
print((test_df.count(), len(test_df.columns)))
# Show the merged DataFrame
test_df.show()

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.sql.functions import sum, when, col

# Calculate specificity and sensitivity by race_category
metrics_by_race = df.groupBy("race_category").agg(
    (sum(when((col("prediction") == 1) & (col("non_emerg_binary") == 1), 1))
     / sum(when(col("non_emerg_binary") == 1, 1))).alias("sensitivity"),
    (sum(when((col("prediction") == 0) & (col("non_emerg_binary") == 0), 1))
     / sum(when(col("non_emerg_binary") == 0, 1))).alias("specificity")
)

# Show the results
metrics_by_race.show()

# COMMAND ----------

import numpy as np
from scipy import stats

def calculate_sensitivity_specificity(df, prediction_col, truth_col, race_category):
    true_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()
    false_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    true_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    false_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    return sensitivity, specificity

def calculate_mean_confidence_interval(data):
    mean = np.mean(data)
    if len(data) > 1:
        confidence_interval = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
    else:
        confidence_interval = (mean, mean)
    return mean, confidence_interval

# Example usage
race_categories = ['minority', 'missing', 'White', 'Black', 'Hispanic']  # Replace with your actual race categories

num_bootstraps = 1000

results = {}

for race_category in race_categories:
    sensitivities = []
    specificities = []

    for _ in range(num_bootstraps):
        bootstrap_sample = df.sample(withReplacement=True, fraction=0.05)

        sensitivity, specificity = calculate_sensitivity_specificity(bootstrap_sample, 'prediction', 'non_emerg_binary', race_category)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    mean_sensitivity, ci_sensitivity = calculate_mean_confidence_interval(sensitivities)
    mean_specificity, ci_specificity = calculate_mean_confidence_interval(specificities)

    results[race_category] = {
        'mean_sensitivity': mean_sensitivity,
        'ci_sensitivity': ci_sensitivity,
        'mean_specificity': mean_specificity,
        'ci_specificity': ci_specificity
    }

# Print the results
for race_category, metrics in results.items():
    print(f"Race Category: {race_category}")
    print(f"Mean Sensitivity: {metrics['mean_sensitivity']}")
    print(f"Sensitivity CI: {metrics['ci_sensitivity']}")
    print(f"Mean Specificity: {metrics['mean_specificity']}")
    print(f"Specificity CI: {metrics['ci_specificity']}")
    print("-------------")

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.test_xg_boost_stage2_allcause_new1")
df = spark.table("dua_058828_spa240.stage2_random_sample_5million")
df = df.select("beneID","state","race")
df = df.withColumn(
    "race_category",
    when(col("race") == "white", "White")
    .when(col("race") == "missing", "missing")
    .when(col("race") == "black", "Black")
    .when(col("race") == "hispanic", "Hispanic")
    .otherwise("minority")
)

print((test_df.count(), len(test_df.columns)))
print((df.count(), len(df.columns)))

# Perform the left merge on 'beneID' and 'state' columns
print((test_df.count(), len(test_df.columns)))
df_final = test_df.join(df, on=['beneID', 'state'], how='left')
print((df_final.count(), len(df_final.columns)))
# Show the merged DataFrame
df_final.show()

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.sql.functions import sum, when, col


# Calculate specificity and sensitivity by race_category
metrics_by_race = df_final.groupBy("race_category").agg(
    (sum(when((col("prediction") == 1) & (col("all_cause_binary") == 1), 1))
     / sum(when(col("all_cause_binary") == 1, 1))).alias("sensitivity"),
    (sum(when((col("prediction") == 0) & (col("all_cause_binary") == 0), 1))
     / sum(when(col("all_cause_binary") == 0, 1))).alias("specificity")
)

# Show the results
metrics_by_race.show()

# COMMAND ----------

import numpy as np
from scipy import stats

df =df_final

def calculate_sensitivity_specificity(df, prediction_col, truth_col, race_category):
    true_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()
    false_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    true_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    false_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    
    return sensitivity, specificity

def calculate_mean_confidence_interval(data):
    mean = np.mean(data)
    if len(data) > 1:
        confidence_interval = stats.norm.interval(0.95, loc=mean, scale=stats.sem(data))
    else:
        confidence_interval = (mean, mean)
    return mean, confidence_interval

# Example usage
race_categories = ['minority', 'missing', 'White', 'Black', 'Hispanic']  # Replace with your actual race categories

num_bootstraps = 1000

results = {}

for race_category in race_categories:
    sensitivities = []
    specificities = []

    for _ in range(num_bootstraps):
        bootstrap_sample = df.sample(withReplacement=True, fraction=0.05)

        sensitivity, specificity = calculate_sensitivity_specificity(bootstrap_sample, 'prediction', 'all_cause_binary', race_category)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    mean_sensitivity, ci_sensitivity = calculate_mean_confidence_interval(sensitivities)
    mean_specificity, ci_specificity = calculate_mean_confidence_interval(specificities)

    results[race_category] = {
        'mean_sensitivity': mean_sensitivity,
        'ci_sensitivity': ci_sensitivity,
        'mean_specificity': mean_specificity,
        'ci_specificity': ci_specificity
    }

# Print the results
for race_category, metrics in results.items():
    print(f"Race Category: {race_category}")
    print(f"Mean Sensitivity: {metrics['mean_sensitivity']}")
    print(f"Sensitivity CI: {metrics['ci_sensitivity']}")
    print(f"Mean Specificity: {metrics['mean_specificity']}")
    print(f"Specificity CI: {metrics['ci_specificity']}")
    print("-------------")

# COMMAND ----------

