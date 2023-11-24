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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.linalg import Vectors
import numpy as np

test_df = spark.table("dua_058828_spa240.test_xg_non_emerg_sensitivity_adults")

tp = test_df.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 1.0)).count()
tn = test_df.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 0.0)).count()
fp = test_df.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 0.0)).count()
fn = test_df.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 1.0)).count()

precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

print(f1_score)

iteration_levels = [1000]
fraction = 0.05

def bootstrap_metrics_f1(df, n_iterations, fraction):
    f1s = []
    for _ in range(n_iterations):
        # Create a bootstrap sample with replacement
        bootstrap_sample = df.sample(withReplacement=True, fraction=fraction)
        
 
        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        
        tp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 1.0)).count()
        tn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 0.0)).count()
        fp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 0.0)).count()
        fn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 1.0)).count()

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
                
        f1s.append(f1_score)    
    return f1s
  
# Perform bootstrap analysis for each iteration level for the test set
test_f1_list = []
test_ci_fi_list = []

for n_iterations in iteration_levels:
    f1s = bootstrap_metrics_f1(test_df, n_iterations, fraction)
    test_f1_list.append(np.mean(f1s))
    test_ci_fi_list.append(np.percentile(f1s, [2.5, 97.5]))

print(test_f1_list)
print(test_ci_fi_list)

# COMMAND ----------

