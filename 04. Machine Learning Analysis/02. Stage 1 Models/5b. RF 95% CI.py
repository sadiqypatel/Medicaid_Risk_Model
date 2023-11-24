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

def calculate_mcc(tn, fp, fn, tp):
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return numerator / denominator if denominator != 0.0 else 0.0
  
def calculate_mean_and_ci(metric_values):
    mean_value = np.mean(metric_values)
    ci_value = np.percentile(metric_values, [2.5, 97.5])
    return mean_value, ci_value

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.linalg import Vectors
import numpy as np

def bootstrap_metrics(df, n_iterations, fraction):
    aucs = []
    mccs = []
    accuracies = []
    sensitivities = []
    specificities = []
    npvs = []
    ppvs = []
    for _ in range(n_iterations):
        # Create a bootstrap sample with replacement
        bootstrap_sample = df.sample(withReplacement=True, fraction=fraction)
        
        auc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='loseCoverage', metricName='areaUnderROC')
        auc = auc_evaluator.evaluate(bootstrap_sample)
        
        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        
        tp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('loseCoverage') == 1.0)).count()
        tn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('loseCoverage') == 0.0)).count()
        fp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('loseCoverage') == 0.0)).count()
        fn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('loseCoverage') == 1.0)).count()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
        specificity = tn / (tn + fp) if (tn + fp) > 0 else None
        ppv = tp / (tp + fp) if (tp + fp) > 0 else None
        npv = tn / (tn + fn) if (tn + fn) > 0 else None
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
        mcc = calculate_mcc(tn, fp, fn, tp)
                
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        npvs.append(npv)
        ppvs.append(ppv)
        mccs.append(mcc)
        accuracies.append(accuracy)
        aucs.append(auc)
        
    return aucs, mccs, accuracies, sensitivities, specificities, npvs, ppvs

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

test_df = spark.table("dua_058828_spa240.test_rf_stage1")

# COMMAND ----------

#iteration_levels = [1]
iteration_levels = [1000]
fraction = 0.05

# COMMAND ----------

# Perform bootstrap analysis for each iteration level for the test set
test_aucs_list = []
test_mccs_list = []
test_accuracies_list = []
test_ppvs_list = []
test_npvs_list = []
test_sensitivities_list = []
test_specificities_list = []

test_ci_aucs_list = []
test_ci_mccs_list = []
test_ci_accuracies_list = []
test_ci_ppvs_list = []
test_ci_npvs_list = []
test_ci_sensitivities_list = []
test_ci_specificities_list = []

for n_iterations in iteration_levels:
    aucs, mccs, accuracies, sensitivities, specificities, npvs, ppvs = bootstrap_metrics(test_df, n_iterations, fraction)
    test_aucs_list.append(np.mean(aucs))
    test_mccs_list.append(np.mean(mccs))
    test_accuracies_list.append(np.mean(accuracies))
    test_sensitivities_list.append(np.mean(sensitivities))
    test_specificities_list.append(np.mean(specificities))
    test_npvs_list.append(np.mean(npvs))
    test_ppvs_list.append(np.mean(ppvs))
    
    test_ci_aucs_list.append(np.percentile(aucs, [2.5, 97.5]))
    test_ci_mccs_list.append(np.percentile(mccs, [2.5, 97.5]))
    test_ci_accuracies_list.append(np.percentile(accuracies, [2.5, 97.5]))
    test_ci_sensitivities_list.append(np.percentile(sensitivities, [2.5, 97.5]))
    test_ci_specificities_list.append(np.percentile(specificities, [2.5, 97.5]))
    test_ci_npvs_list.append(np.percentile(npvs, [2.5, 97.5]))
    test_ci_ppvs_list.append(np.percentile(ppvs, [2.5, 97.5]))

print(test_aucs_list)
print(test_mccs_list)
print(test_accuracies_list)    
print(test_sensitivities_list)
print(test_specificities_list)
print(test_npvs_list)
print(test_ppvs_list)

print(test_ci_aucs_list)
print(test_ci_mccs_list)
print(test_ci_accuracies_list)    
print(test_ci_sensitivities_list)
print(test_ci_specificities_list)
print(test_ci_npvs_list)
print(test_ci_ppvs_list)