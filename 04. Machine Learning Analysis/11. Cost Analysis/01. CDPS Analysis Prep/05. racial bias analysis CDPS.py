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

df = spark.table("dua_058828_spa240.stage2_cost_analysis_sample")

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
test_disabled = spark.table("dua_058828_spa240.paper1_stage2_cdps_disabled_2M_pred")
test_disabled = test_disabled.withColumnRenamed("Recipient_ID", "beneID").withColumnRenamed("DGNS_CD_1", "DIAG_ID")
test_disabled = test_disabled.withColumn("diff", expr("(prediction - total_cost)"))
print((test_disabled.count(), len(test_disabled.columns)))
df_disabled = test_disabled.join(df, on=['beneID', 'state'], how='left')
print((df_disabled.count(), len(df_disabled.columns)))

# Perform the left merge on 'beneID' and 'state' columns
test_adults = spark.table("dua_058828_spa240.paper1_stage2_cdps_adults_2M_pred")
test_adults = test_adults.withColumnRenamed("Recipient_ID", "beneID").withColumnRenamed("DGNS_CD_1", "DIAG_ID")
test_adults = test_adults.withColumn("diff", expr("(prediction - total_cost)"))
print((test_adults.count(), len(test_adults.columns)))
df_adults = test_adults.join(df, on=['beneID', 'state'], how='left')
print((df_adults.count(), len(df_adults.columns)))

# Perform the left merge on 'beneID' and 'state' columns
test_kids = spark.table("dua_058828_spa240.paper1_stage2_cdps_kids_2M_pred")
test_kids = test_kids.withColumnRenamed("Recipient_ID", "beneID").withColumnRenamed("DGNS_CD_1", "DIAG_ID")
test_kids = test_kids.withColumn("diff", expr("(prediction - total_cost)"))
print((test_kids.count(), len(test_kids.columns)))
df_kids = test_kids.join(df, on=['beneID', 'state'], how='left')
print((df_kids.count(), len(df_kids.columns)))

# COMMAND ----------

from pyspark.sql.functions import mean
import random
import numpy as np

def calculate_mean_cost_by_race(df):
    #mean_cost_df = df.groupBy("race_category").agg(mean("prediction").alias("mean_cost"))
    mean_diff_df = df.groupBy("race_category").agg(mean("diff").alias("mean_diff"))
    return mean_diff_df

def calculate_confidence_interval(means_dict, confidence_level=0.95):
    result = {}

    for race, means in means_dict.items():
        num_samples = len(means)
        mean_value = np.mean(means)
        std_error = np.std(means) / np.sqrt(num_samples)
        z_score = 1.96  # Z-score for 95% confidence interval

        lower_bound = mean_value - z_score * std_error
        upper_bound = mean_value + z_score * std_error

        result[race] = {"mean": mean_value, "lower_bound": lower_bound, "upper_bound": upper_bound}

    return result
  
def perform_bootstrap(df, race_category_column, prediction_column, sample_fraction, num_iterations):
    race_categories = df.select(race_category_column).distinct().rdd.flatMap(lambda x: x).collect()
    means_by_race = {race: [] for race in race_categories}

    for _ in range(num_iterations):
        bootstrap_sample = df.sample(withReplacement=True, fraction=sample_fraction, seed=random.randint(1, 1000))
        race_sample_means = bootstrap_sample.groupBy(race_category_column).agg(mean(prediction_column).alias("mean")).collect()

        for row in race_sample_means:
            race = row[race_category_column]
            mean_value = row["mean"]
            means_by_race[race].append(mean_value)

    return means_by_race

# COMMAND ----------

mean_diff_df = calculate_mean_cost_by_race(df_disabled)
mean_diff_df.show()

mean_diff_df = calculate_mean_cost_by_race(df_adults)
mean_diff_df.show()

mean_diff_df = calculate_mean_cost_by_race(df_kids)
mean_diff_df.show()

# COMMAND ----------

bootstrap_means = perform_bootstrap(df_adults, "race_category", "diff", sample_fraction=0.25, num_iterations=500)
#print(step1)

confidence_intervals = calculate_confidence_interval(bootstrap_means)

for race, interval in confidence_intervals.items():
    print(race)
    print("Mean:", interval["mean"])
    print("Lower Bound:", interval["lower_bound"])
    print("Upper Bound:", interval["upper_bound"])
    print()

# COMMAND ----------

bootstrap_means = perform_bootstrap(df_kids, "race_category", "diff", sample_fraction=0.25, num_iterations=1000)
#print(step1)

confidence_intervals = calculate_confidence_interval(bootstrap_means)

for race, interval in confidence_intervals.items():
    print(race)
    print("Mean:", interval["mean"])
    print("Lower Bound:", interval["lower_bound"])
    print("Upper Bound:", interval["upper_bound"])
    print()

# COMMAND ----------

bootstrap_means = perform_bootstrap(df_disabled, "race_category", "diff", sample_fraction=0.25, num_iterations=1000)
#print(step1)

confidence_intervals = calculate_confidence_interval(bootstrap_means)

for race, interval in confidence_intervals.items():
    print(race)
    print("Mean:", interval["mean"])
    print("Lower Bound:", interval["lower_bound"])
    print("Upper Bound:", interval["upper_bound"])
    print()