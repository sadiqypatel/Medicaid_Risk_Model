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
from sparkdl.xgboost import XgboostRegressor
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.linalg import Vectors
import numpy as np

def bootstrap_metrics(df, n_iterations, fraction):
    mean_costs = []
    r2_values = []
    for _ in range(n_iterations):
        bootstrap_sample = df.sample(withReplacement=True, fraction=fraction)
        mean_cost = bootstrap_sample.select(F.mean("total_cost")).collect()[0][0]
        mean_costs.append(mean_cost)
  
        evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
        r2 = evaluator.evaluate(bootstrap_sample)
        r2_values.append(r2)
        
    return mean_costs, r2_values

# COMMAND ----------

iteration_levels = [1000]
fraction = 0.10

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.test_cost_baseline")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_costs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    test_mean_cost = np.mean(mean_costs)
    test_mean_r2 = np.mean(r2_values)
    
    test_cost_ci = np.percentile(mean_costs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(test_mean_cost)
print(test_cost_ci)

print(test_mean_r2)
print(test_r2_ci)

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.test_cost_no_SDOH")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_costs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    test_mean_cost = np.mean(mean_costs)
    test_mean_r2 = np.mean(r2_values)
    
    test_cost_ci = np.percentile(mean_costs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(test_mean_cost)
print(test_cost_ci)

print(test_mean_r2)
print(test_r2_ci)

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.test_cost_SDOH_only")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_costs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    test_mean_cost = np.mean(mean_costs)
    test_mean_r2 = np.mean(r2_values)
    
    test_cost_ci = np.percentile(mean_costs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(test_mean_cost)
print(test_cost_ci)

print(test_mean_r2)
print(test_r2_ci)

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.test_cost_all_features")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_costs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    test_mean_cost = np.mean(mean_costs)
    test_mean_r2 = np.mean(r2_values)
    
    test_cost_ci = np.percentile(mean_costs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(test_mean_cost)
print(test_cost_ci)

print(test_mean_r2)
print(test_r2_ci)