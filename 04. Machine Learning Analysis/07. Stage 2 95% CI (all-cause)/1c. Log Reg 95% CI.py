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
        
        auc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='all_cause_binary', metricName='areaUnderROC')
        auc = auc_evaluator.evaluate(bootstrap_sample)
        
        # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
        
        tp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 1.0)).count()
        tn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 0.0)).count()
        fp = bootstrap_sample.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 0.0)).count()
        fn = bootstrap_sample.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 1.0)).count()

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

train_df = spark.table("dua_058828_spa240.stage2_baseline_all_cause_train_log_reg")
test_df = spark.table("dua_058828_spa240.stage2_baseline_all_cause_test_log_reg")

train_df.printSchema()
test_df.printSchema()

train_df.show(10)

# COMMAND ----------

# Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)

tp = train_df.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 1.0)).count()
tn = train_df.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 0.0)).count()
fp = train_df.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 0.0)).count()
fn = train_df.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 1.0)).count()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
specificity = tn / (tn + fp) if (tn + fp) > 0 else None
ppv = tp / (tp + fp) if (tp + fp) > 0 else None
npv = tn / (tn + fn) if (tn + fn) > 0 else None
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
mcc = calculate_mcc(tn, fp, fn, tp)

print(sensitivity)
print(specificity)
print(npv)
print(ppv)

# COMMAND ----------

#iteration_levels = [1]
iteration_levels = [1000]
fraction = 0.05

# COMMAND ----------

# Perform bootstrap analysis for each iteration level for the training set
train_aucs_list = []
train_mccs_list = []
train_accuracies_list = []
train_ppvs_list = []
train_npvs_list = []
train_sensitivities_list = []
train_specificities_list = []

train_ci_aucs_list = []
train_ci_mccs_list = []
train_ci_accuracies_list = []
train_ci_ppvs_list = []
train_ci_npvs_list = []
train_ci_sensitivities_list = []
train_ci_specificities_list = []

for n_iterations in iteration_levels:
    aucs, mccs, accuracies, sensitivities, specificities, npvs, ppvs = bootstrap_metrics(train_df, n_iterations, fraction)
    train_aucs_list.append(np.mean(aucs))
    train_mccs_list.append(np.mean(mccs))
    train_accuracies_list.append(np.mean(accuracies))
    train_sensitivities_list.append(np.mean(sensitivities))
    train_specificities_list.append(np.mean(specificities))
    train_npvs_list.append(np.mean(npvs))
    train_ppvs_list.append(np.mean(ppvs))
    
    train_ci_aucs_list.append(np.percentile(aucs, [2.5, 97.5]))
    train_ci_mccs_list.append(np.percentile(mccs, [2.5, 97.5]))
    train_ci_accuracies_list.append(np.percentile(accuracies, [2.5, 97.5]))
    train_ci_sensitivities_list.append(np.percentile(sensitivities, [2.5, 97.5]))
    train_ci_specificities_list.append(np.percentile(specificities, [2.5, 97.5]))
    train_ci_npvs_list.append(np.percentile(npvs, [2.5, 97.5]))
    train_ci_ppvs_list.append(np.percentile(ppvs, [2.5, 97.5]))

print(train_aucs_list)
print(train_mccs_list)
print(train_accuracies_list)    
print(train_sensitivities_list)
print(train_specificities_list)
print(train_npvs_list)
print(train_ppvs_list)

print(train_ci_aucs_list)
print(train_ci_mccs_list)
print(train_ci_accuracies_list)    
print(train_ci_sensitivities_list)
print(train_ci_specificities_list)
print(train_ci_npvs_list)
print(train_ci_ppvs_list)

# COMMAND ----------

# train_AUC: 0.694
# train_Accuracy: 0.769
# train_MCC: 0.211
# train_Sensitivity: 0.114
# train_PPV: 0.688
# train_Specificity: 0.983
# train_NPV: 0.773

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

# COMMAND ----------

# test_AUC: 0.695
# test_Accuracy: 0.770
# test_MCC: 0.212
# test_Sensitivity: 0.115
# test_PPV: 0.688
# test_Specificity: 0.983
# test_NPV: 0.773

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_accuracies_list
test_list = test_accuracies_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_accuracies_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_accuracies_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_aucs_list
test_list = test_aucs_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_aucs_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_aucs_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_mccs_list
test_list = test_mccs_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_mccs_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_mccs_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_sensitivities_list
test_list = test_sensitivities_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_sensitivities_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_sensitivities_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_specificities_list
test_list = test_specificities_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_specificities_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_specificities_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_npvs_list
test_list = test_npvs_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_npvs_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_npvs_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Extract the lower and upper bounds for each iteration level
train_list = train_ppvs_list
test_list = test_ppvs_list
train_lower_bounds, train_upper_bounds = zip(*train_ci_ppvs_list)
test_lower_bounds, test_upper_bounds = zip(*test_ci_ppvs_list)

# Plot the mean MCC values for the training and test sets
plt.figure(figsize=(10, 6))
plt.plot(iteration_levels, train_list, marker='o', label='Train')
plt.fill_between(iteration_levels, train_lower_bounds, train_upper_bounds, color='yellow', alpha=0.5, label='Train 95% CI')

plt.plot(iteration_levels, test_list, marker='o', label='Test')
plt.fill_between(iteration_levels, test_lower_bounds, test_upper_bounds, color='blue', alpha=0.5, label='Test 95% CI')

plt.xlabel('Number of Iterations')
plt.ylabel('Mean')
#plt.title('Bootstrap Analysis: Accuracy')
plt.ylim(0, 1.0)  # Set the y-axis range to [0, 1]
plt.legend()
plt.grid()

# COMMAND ----------

