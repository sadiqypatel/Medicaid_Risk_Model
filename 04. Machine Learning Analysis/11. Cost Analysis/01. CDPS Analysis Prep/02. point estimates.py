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
    mean_preds = []
    mean_actuals = []
    mean_diffs = []
    r2_values = []
    for _ in range(n_iterations):
        bootstrap_sample = df.sample(withReplacement=False, fraction=fraction)
        mean_actual = bootstrap_sample.select(F.mean("total_cost")).collect()[0][0]
        mean_pred = bootstrap_sample.select(F.mean("prediction")).collect()[0][0]
        #bootstrap_sample = bootstrap_sample.withColumn("diff", abs(col("total_cost") - col("prediction")))
        bootstrap_sample = bootstrap_sample.withColumn("diff", expr("abs(total_cost - prediction)"))
        mean_diff = bootstrap_sample.select(F.mean("diff")).collect()[0][0]
        mean_actuals.append(mean_actual)  
        mean_preds.append(mean_pred)  
        mean_diffs.append(mean_diff)
        evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
        r2 = evaluator.evaluate(bootstrap_sample)
        r2_values.append(r2)
        
    return mean_actuals, mean_preds, mean_diffs, r2_values

# COMMAND ----------

iteration_levels = [1]
fraction = 1.0

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.paper1_stage2_cdps_kids_2M_pred")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_actuals, mean_preds, mean_diffs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    mean_actuals_test = np.mean(mean_actuals)
    mean_preds_test = np.mean(mean_preds)
    mean_diffs_test = np.mean(mean_diffs)
    mean_r2_values_test = np.mean(r2_values)
    
    test_actuals_ci = np.percentile(mean_actuals, [2.5, 97.5])
    test_preds_ci = np.percentile(mean_preds, [2.5, 97.5])
    test_diffs_ci = np.percentile(mean_diffs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(mean_actuals_test)
print(test_actuals_ci)

print(mean_preds_test)
print(test_preds_ci)

print(mean_diffs_test)
print(test_diffs_ci)

print(mean_r2_values_test)
print(test_r2_ci)

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.paper1_stage2_cdps_adults_2M_pred")
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_actuals, mean_preds, mean_diffs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    mean_actuals_test = np.mean(mean_actuals)
    mean_preds_test = np.mean(mean_preds)
    mean_diffs_test = np.mean(mean_diffs)
    mean_r2_values_test = np.mean(r2_values)
    
    test_actuals_ci = np.percentile(mean_actuals, [2.5, 97.5])
    test_preds_ci = np.percentile(mean_preds, [2.5, 97.5])
    test_diffs_ci = np.percentile(mean_diffs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(mean_actuals_test)
print(test_actuals_ci)

print(mean_preds_test)
print(test_preds_ci)

print(mean_diffs_test)
print(test_diffs_ci)

print(mean_r2_values_test)
print(test_r2_ci)

# COMMAND ----------

test_df = spark.table("dua_058828_spa240.paper1_stage2_cdps_disabled_2M_pred")
print(test_df.count())
#test_df.printSchema()

for n_iterations in iteration_levels:
    mean_actuals, mean_preds, mean_diffs, r2_values = bootstrap_metrics(test_df, n_iterations, fraction)
    mean_actuals_test = np.mean(mean_actuals)
    mean_preds_test = np.mean(mean_preds)
    mean_diffs_test = np.mean(mean_diffs)
    mean_r2_values_test = np.mean(r2_values)
    
    test_actuals_ci = np.percentile(mean_actuals, [2.5, 97.5])
    test_preds_ci = np.percentile(mean_preds, [2.5, 97.5])
    test_diffs_ci = np.percentile(mean_diffs, [2.5, 97.5])
    test_r2_ci = np.percentile(r2_values, [2.5, 97.5])

print(mean_actuals_test)
print(test_actuals_ci)

print(mean_preds_test)
print(test_preds_ci)

print(mean_diffs_test)
print(test_diffs_ci)

print(mean_r2_values_test)
print(test_r2_ci)

# COMMAND ----------

