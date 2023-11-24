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
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from sklearn.model_selection import KFold
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from sklearn.metrics import matthews_corrcoef
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from sklearn.model_selection import KFold
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col
from sklearn.metrics import matthews_corrcoef
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import matthews_corrcoef
import warnings
import sparkdl.xgboost
from sparkdl.xgboost import XgboostClassifier

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_random_sample_5million_base_vector_assembler")
print(df.count())
df = df.withColumn("all_cause_binary", when(col("all_cause_acute_post") > 0, 1).otherwise(0))
df.groupBy("all_cause_binary").count().orderBy(col("count").desc()).show()

# COMMAND ----------

#Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction =  df.count() / df.count()

#fraction = 25000 / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

# Continue with your analysis using the sampled DataFrame 'sampled_df'

# COMMAND ----------

sampled_df = sampled_df.select("beneID", "state", "features", "all_cause_binary")
print(sampled_df.count())

# COMMAND ----------

# Split the data into training and test sets
train_df, test_df = sampled_df.randomSplit([0.8, 0.2], seed=1234)

print(train_df.count())
print(test_df.count())
#display(train_df)

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

# # # Perform an additional 80/20 train/validation split on the training set
train_val_ratio = 0.8
train_sub_df, val_df = train_df.randomSplit([train_val_ratio, 1 - train_val_ratio], seed=42)

# COMMAND ----------

from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# # # Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/8. stage 2 base model (cdps equiv)/7. XGboost all-cause")
mlflow.set_tracking_uri("databricks")

# Create an instance of the XGBoostClassifier
xgb_classifier = XgboostClassifier(featuresCol="features", labelCol="all_cause_binary")

hyperparameters = [
    {'learning_rate': 0.005,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 22, 'reg_alpha': 0.5,  'reg_lambda': 0.5,  'gamma': 0, 'colsample_bylevel': 0.5}
]


# Create an instance of the BinaryClassificationEvaluator
auc_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",  # The column containing raw predictions (e.g., rawPrediction)
    labelCol="all_cause_binary",  # The column containing true labels (e.g., label)
    metricName="areaUnderROC"  # The metric to evaluate (e.g., area under the ROC curve)
)

# Create an instance of the MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="all_cause_binary",  # The column containing true labels (e.g., label)
    predictionCol="prediction",  # The column containing predicted labels (e.g., prediction)
    metricName="accuracy"  # The metric to evaluate (e.g., accuracy)
)

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run()

# # # # Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/8. stage 2 base model (cdps equiv)/7. XGboost all-cause")
mlflow.set_tracking_uri("databricks")

# # # Initialize variables to keep track of the best model and its metrics
best_model = None
best_metrics = None
best_mcc = -1.0
best_run_id = None

# Loop through each hyperparameter combination in the param_grid
for params in hyperparameters:
    # Create an instance of the XGBoostClassifier and set the hyperparameters
    xgb = XgboostClassifier(featuresCol="features", labelCol="all_cause_binary", missing=0.0, **params)
    #print(xgb_model)
    model = xgb.fit(train_sub_df)
    predictions = model.transform(val_df)
    
        # Evaluate the performance of the best model
    auc = auc_evaluator.evaluate(predictions)
    tp = predictions.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 1.0)).count()
    tn = predictions.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 0.0)).count()
    fp = predictions.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 0.0)).count()
    fn = predictions.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 1.0)).count()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0.0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0.0 else 0.0
    mcc = calculate_mcc(tn, fp, fn, tp)    
    
    learning_rate = params['learning_rate']
    subsample = params['subsample']
    min_child_weight = params['min_child_weight']
    colsample_bytree = params['colsample_bytree']
    max_depth = params['max_depth']
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']
    gamma = params['gamma']
    colsample_bylevel = params['colsample_bylevel']

    # Log the hyperparameters and metrics to MLflow
    with mlflow.start_run():
        # Log hyperparameters        
        for key, value in params.items(): 
            mlflow.log_param(key, value)
   
        # Log metrics
        mlflow.log_metrics({
        "AUC": auc,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "PPV": ppv,
        "Specificity": specificity,
        "MCC": mcc,
        "NPV": npv
        })
        
        # Update the best model and metrics if the current model is better
        if mcc > best_mcc:
            best_model = model
            best_mcc = mcc
            best_run_id = mlflow.active_run().info.run_id
            best_run_info = mlflow.get_run(best_run_id)
            
            best_params = {
            "best_learning_rate": learning_rate,
            "best_subsample": subsample,
            "best_min_child_weight": min_child_weight,
            "best_colsample_bytree": colsample_bytree,
            "best_max_depth": max_depth,
            "best_reg_alpha": reg_alpha,
            "best_reg_lambda": reg_lambda,
            "best_gamma": gamma,
            "best_colsample_bylevel": colsample_bylevel
            }
            
            best_metrics = {
            "AUC": auc,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "PPV": ppv,
            "Specificity": specificity,
            "MCC": mcc,
            "NPV": npv
                }

# # Capture the current run ID
run_id = best_run_id
run_info = best_run_info

# # Print the metric output for the best model
for metric_name, metric_value in best_metrics.items():
    print(f"{metric_name}: {metric_value:.5f}")

# COMMAND ----------

print(run_info)

# COMMAND ----------

best_hyperparams = best_run_info.data.params
print(best_hyperparams)

# COMMAND ----------

import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

if mlflow.active_run():
    mlflow.end_run()
    
# Retrieve the run information from MLflow
run_info = mlflow.get_run(run_id)
#print(run_info)
# # # # Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/8. stage 2 base model (cdps equiv)/7. XGboost all-cause")
mlflow.set_tracking_uri("databricks")
        
# # Extract the hyperparameters from the run information
best_hyperparams = run_info.data.params
print(best_hyperparams)
best_hyperparams = {k: float(v) for k, v in best_hyperparams.items()}
best_hyperparams['max_depth'] = int(float(best_hyperparams['max_depth']))
print(best_hyperparams)

# Create a new instance of the estimator with the same hyperparameters as the best_model
final_estimator = XgboostClassifier(featuresCol="features", labelCol='all_cause_binary', missing=0.0, **best_hyperparams)

# Fit the final model to the full training dataset (train_df)
final_model = final_estimator.fit(train_df)

# Make predictions on the full training dataset
train_predictions = final_model.transform(train_df)

# Make predictions on the test dataset
test_predictions = final_model.transform(test_df)

def calculate_metrics(predictions, cohort_tag):
    # Calculate AUC
    auc = auc_evaluator.evaluate(predictions)
    
    tp = predictions.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 1.0)).count()
    tn = predictions.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 0.0)).count()
    fp = predictions.filter((col('prediction') == 1.0) & (col('all_cause_binary') == 0.0)).count()
    fn = predictions.filter((col('prediction') == 0.0) & (col('all_cause_binary') == 1.0)).count()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    mcc = calculate_mcc(tn, fp, fn, tp)
    
    # Log in MLFLOW
    mlflow.log_metric(f"{cohort_tag}_AUC", auc)
    mlflow.log_metric(f"{cohort_tag}_Accuracy", accuracy)
    mlflow.log_metric(f"{cohort_tag}_Sensitivity", sensitivity)
    mlflow.log_metric(f"{cohort_tag}_PPV", ppv)
    mlflow.log_metric(f"{cohort_tag}_Specificity", specificity)
    mlflow.log_metric(f"{cohort_tag}_MCC", mcc)
    mlflow.log_metric(f"{cohort_tag}_NPV", npv)
    
    # Print all metrics
    print(f"{cohort_tag}_AUC: {auc:.3f}")
    print(f"{cohort_tag}_Accuracy: {accuracy:.3f}")
    print(f"{cohort_tag}_MCC: {mcc:.3f}")
    print(f"{cohort_tag}_Sensitivity: {sensitivity:.3f}")
    print(f"{cohort_tag}_PPV: {ppv:.3f}")
    print(f"{cohort_tag}_Specificity: {specificity:.3f}")
    print(f"{cohort_tag}_NPV: {npv:.3f}")

# COMMAND ----------

# Start an MLflow run to log metrics
with mlflow.start_run(run_name="Final_Model"):
    # Calculate and log metrics on the full training set
    calculate_metrics(train_predictions, cohort_tag="train")

    # Calculate and log metrics on the test set
    calculate_metrics(test_predictions, cohort_tag="test")

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT

# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])

# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
train_logReg_withReg = train_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
train_logReg_withReg = train_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'all_cause_binary')

# Display the selected columns
train_logReg_withReg.show()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT

# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])

# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
test_logReg_withReg = test_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
test_logReg_withReg = test_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'all_cause_binary')

# Display the selected columns
test_logReg_withReg.show(50)

# COMMAND ----------

train_logReg_withReg.write.saveAsTable("dua_058828_spa240.train_xg_boost_stage2_allcause_base", mode="overwrite")
test_logReg_withReg.write.saveAsTable("dua_058828_spa240.test_xg_boost_stage2_allcause_base", mode="overwrite")