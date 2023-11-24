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

test_df = spark.table("dua_058828_spa240.test_xg_boost_stage2_allcause_new1")
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
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# If you have different column names, replace them accordingly
prediction_col = "prediction"
true_labels_col = "all_cause_binary"

# Convert "prediction" and "all_cause_binary" columns to integer
df = df.withColumn("prediction", col("prediction").cast("int"))
df = df.withColumn("all_cause_binary", col("all_cause_binary").cast("int"))

tp_fp_tn_fn_by_race = (
    df.groupBy("race_category")
    .agg(
        sum(when((col(prediction_col) == 1) & (col(true_labels_col) == 1), 1).otherwise(0)).alias("tp"),
        sum(when((col(prediction_col) == 1) & (col(true_labels_col) == 0), 1).otherwise(0)).alias("fp"),
        sum(when((col(prediction_col) == 0) & (col(true_labels_col) == 0), 1).otherwise(0)).alias("tn"),
        sum(when((col(prediction_col) == 0) & (col(true_labels_col) == 1), 1).otherwise(0)).alias("fn")
    )
)

# Calculate precision, recall, and F1 score
metrics_by_race = (
    tp_fp_tn_fn_by_race
    .withColumn(
        "precision",
        col("tp") / (col("tp") + col("fp"))
    )
    .withColumn(
        "recall",
        col("tp") / (col("tp") + col("fn"))
    )
    .withColumn(
        "f1_score",
        2 * (col("precision") * col("recall")) / (col("precision") + col("recall")).alias("f1_score")
    )
    .select("race_category", "precision", "recall", "f1_score")
)

# Show the results
metrics_by_race.show()

# COMMAND ----------

import numpy as np
from scipy import stats

def calculate_f1(df, prediction_col, truth_col, race_category):
    true_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()
    false_positive = df.filter((df[prediction_col] == 1) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    true_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 0) & (df['race_category'] == race_category)).count()
    false_negative = df.filter((df[prediction_col] == 0) & (df[truth_col] == 1) & (df['race_category'] == race_category)).count()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = (2 * precision * recall) / (precision + recall)
       
    return f1_score

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
    f1_scores = []

    for _ in range(num_bootstraps):
        bootstrap_sample = df.sample(withReplacement=True, fraction=0.05)

        f1_score = calculate_f1(bootstrap_sample, 'prediction', 'all_cause_binary', race_category)
        f1_scores.append(f1_score)

    mean_f1, ci_f1 = calculate_mean_confidence_interval(f1_scores)

    results[race_category] = {
        'mean_f1': mean_f1,
        'ci_f1': ci_f1
    }

# Print the results
for race_category, metrics in results.items():
    print(f"Race Category: {race_category}")
    print(f"Mean F1: {metrics['mean_f1']}")
    print(f"F1 CI: {metrics['ci_f1']}")
    print("-------------")

# COMMAND ----------

