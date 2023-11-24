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
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import sparkdl.xgboost
from sparkdl.xgboost import XgboostClassifier
#from pyspark.ml.wrapper import JavaWrapper
#from xgboost4j_spark.ml.dmlc.xgboost4j.scala.spark import XGBoostClassifier

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage1_sample")
print(df.count())
df = df.withColumn("cov2016yes", when(col("2016months") > 0, 1).otherwise(0))
df.show(10)

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = 5000000 / df.count()

#fraction = 10000000 / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

# Continue with your analysis using the sampled DataFrame 'sampled_df'

# COMMAND ----------

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
#warnings.filterwarnings('ignore', category=DeprecationWarning)

# Sample data with categorical variables, continuous variable, and binary label "enrolled"
columns = ['ageCat','sex','race','state','houseSize','fedPovLine','speakEnglish','married','UsCitizen','ssi','ssdi','tanf','disabled','enrollMonth','enrollYear','cov2016yes']

# Define transformers and assembler
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in columns[:]]
encoders = [OneHotEncoder(inputCol=c + "_index", outputCol=c + "_onehot") for c in columns[:]]
assembler = VectorAssembler(inputCols=[c + "_onehot" for c in columns[:]], outputCol="features")
#print(features)

# Create pipeline and transform training data
feature_pipeline = Pipeline(stages=indexers + encoders + [assembler])
transformed_df = feature_pipeline.fit(sampled_df).transform(sampled_df)

# Split the data into training and test sets
train_df, test_df = transformed_df.randomSplit([0.8, 0.2], seed=1234)

# Split the data into training and test sets
#transformed_train_data, transformed_test_data = transformed_df.randomSplit([0.8, 0.2], seed=1234)

print(train_df.count())
print(test_df.count())
display(train_df)

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
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/3. stage 1 models/3b. XGboost point estimate")
mlflow.set_tracking_uri("databricks")

# Create an instance of the XGBoostClassifier
xgb_classifier = XgboostClassifier(featuresCol="features", labelCol="loseCoverage")

# Create an instance of the ParamGridBuilder and define the parameter grid
# Define a more comprehensive hyperparameter grid

# param_grid = (
#     ParamGridBuilder()
#     .addGrid(xgb_classifier.learning_rate, [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])  
#     .addGrid(xgb_classifier.subsample, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])   
#     .addGrid(xgb_classifier.min_child_weight, [0, 1, 2, 3, 4, 5])
#     .addGrid(xgb_classifier.colsample_bytree, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     .addGrid(xgb_classifier.max_depth, [3, 6, 9, 12, 15, 18, 21])
#     .addGrid(xgb_classifier.reg_alpha, [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100])
#     .addGrid(xgb_classifier.reg_lambda, [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10, 100])
#     .addGrid(xgb_classifier.gamma, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5])
#     .addGrid(xgb_classifier.colsample_bylevel, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])    
#     .build()
# )

# param_grid = (
#     ParamGridBuilder()
#     .addGrid(xgb_classifier.learning_rate, [0.01, 0.05])  
#     .addGrid(xgb_classifier.subsample, [0.5])   
#     .addGrid(xgb_classifier.min_child_weight, [0, 1])
#     .addGrid(xgb_classifier.colsample_bytree, [0.5])
#     .addGrid(xgb_classifier.max_depth, [5, 10])
#     .addGrid(xgb_classifier.reg_alpha, [0.0, 0.25])
#     .addGrid(xgb_classifier.reg_lambda, [0.0, 0.25])
#     .addGrid(xgb_classifier.gamma, [0.0, 0.25])
#     .addGrid(xgb_classifier.colsample_bylevel, [0.5])          
#     .build()
# )

# hyperparameters = [
#     {'learning_rate': 0.01,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.5,  'reg_lambda': 0.0,  'gamma': 0, 'colsample_bylevel': 0.5},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.25, 'reg_lambda': 0.25, 'gamma': 1, 'colsample_bylevel': 1},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 18, 'reg_alpha': 0.0,  'reg_lambda': 0.5,  'gamma': 1, 'colsample_bylevel': 1},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.5,  'reg_lambda': 0.0,  'gamma': 1, 'colsample_bylevel': 0.5},
#     {'learning_rate': 0.01,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 16, 'reg_alpha': 0.0,  'reg_lambda': 0.5,  'gamma': 1, 'colsample_bylevel': 0.5},
#     {'learning_rate': 0.01,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 18, 'reg_alpha': 0.25, 'reg_lambda': 0.25, 'gamma': 1, 'colsample_bylevel': 1},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.25, 'reg_lambda': 0.5,  'gamma': 0, 'colsample_bylevel': 0.5},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 22, 'reg_alpha': 0.25, 'reg_lambda': 0.5,  'gamma': 1, 'colsample_bylevel': 1},
#     {'learning_rate': 0.005, 'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.25, 'reg_lambda': 0.5,  'gamma': 1, 'colsample_bylevel': 1},
#     {'learning_rate': 0.01,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.0,  'reg_lambda': 0.5,  'gamma': 1, 'colsample_bylevel': 1}
#     # Add more dictionaries for other sets of hyperparameters
# ]

hyperparameters = [
    {'learning_rate': 0.01,  'subsample': 0.5, 'min_child_weight': 0, 'colsample_bytree': 1.0, 'max_depth': 20, 'reg_alpha': 0.5,  'reg_lambda': 0.0,  'gamma': 0, 'colsample_bylevel': 0.5},
]

  
# Create an instance of the BinaryClassificationEvaluator
auc_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",  # The column containing raw predictions (e.g., rawPrediction)
    labelCol="loseCoverage",  # The column containing true labels (e.g., label)
    metricName="areaUnderROC"  # The metric to evaluate (e.g., area under the ROC curve)
)

# Create an instance of the MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="loseCoverage",  # The column containing true labels (e.g., label)
    predictionCol="prediction",  # The column containing predicted labels (e.g., prediction)
    metricName="accuracy"  # The metric to evaluate (e.g., accuracy)
)

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run()

# # # # Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/3. stage 1 models/3b. XGboost point estimate")
mlflow.set_tracking_uri("databricks")

# # # Initialize variables to keep track of the best model and its metrics
best_model = None
best_metrics = None
best_mcc = -1.0
best_run_id = None

# Loop through each hyperparameter combination in the param_grid
for params in hyperparameters:
    # Create an instance of the XGBoostClassifier and set the hyperparameters
    xgb = XgboostClassifier(featuresCol="features", labelCol="loseCoverage", missing=0.0, **params)
    #print(xgb_model)
    model = xgb.fit(train_sub_df)
    predictions = model.transform(val_df)
    
        # Evaluate the performance of the best model
    auc = auc_evaluator.evaluate(predictions)
    tp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 1.0)).count()
    tn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 0.0)).count()
    fp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 0.0)).count()
    fn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 1.0)).count()

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
            
#         # Optionally, you can also log the best model itself to MLflow
#         mlflow.spark.log_model(best_model, "best_model")
#         mlflow.set_tag("model_description", "Best XGBoost based on MCC")
#         mlflow.log_metrics(best_metrics)
#         for key, value in best_params.items():
#             mlflow.log_param(key, value)

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
        
# # Extract the hyperparameters from the run information
best_hyperparams = run_info.data.params
print(best_hyperparams)
best_hyperparams = {k: float(v) for k, v in best_hyperparams.items()}
best_hyperparams['max_depth'] = int(float(best_hyperparams['max_depth']))
print(best_hyperparams)

# Create a new instance of the estimator with the same hyperparameters as the best_model
final_estimator = XgboostClassifier(featuresCol="features", labelCol='loseCoverage', missing=0.0, **best_hyperparams)

# Fit the final model to the full training dataset (train_df)
final_model = final_estimator.fit(train_df)

# Make predictions on the full training dataset
train_predictions = final_model.transform(train_df)

# Make predictions on the test dataset
test_predictions = final_model.transform(test_df)

def calculate_metrics(predictions, cohort_tag):
    # Calculate AUC
    auc = auc_evaluator.evaluate(predictions)
    
    tp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 1.0)).count()
    tn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 0.0)).count()
    fp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 0.0)).count()
    fn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 1.0)).count()

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

from pyspark.ml.classification import LogisticRegressionModel
from typing import List, Dict

def get_pyspark_logistic_regression_feature_importances(
    model: LogisticRegressionModel, 
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Get feature importances for a PySpark logistic regression model, ordered by importance.

    Parameters:
        model (LogisticRegressionModel): A trained PySpark logistic regression model.
        feature_names (List[str]): A list of feature names corresponding to the features used for training.

    Returns:
        Dict[str, float]: A dictionary mapping feature names to their importances, ordered by importance.
    """
    # Get feature importances (coefficients)
    feature_importances = model.coefficients.toArray()

    # Create a dictionary of feature importances
    importance_dict = dict(zip(feature_names, feature_importances))

    # Sort the dictionary by the absolute values of the importances
    sorted_importance_dict = dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    return sorted_importance_dict

# COMMAND ----------

# Start an MLflow run to log metrics
with mlflow.start_run(run_name="Final_Model"):
    # Calculate and log metrics on the full training set
    calculate_metrics(train_predictions, cohort_tag="train")

    # Calculate and log metrics on the test set
    calculate_metrics(test_predictions, cohort_tag="test")

    # Log the final model to MLflow
    mlflow.spark.log_model(final_model, "final_model")

# COMMAND ----------

# display(train_predictions)

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
train_logReg_withReg = train_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'loseCoverage')

# Display the selected columns
train_logReg_withReg.show()

# COMMAND ----------

# display(test_predictions)

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
test_logReg_withReg = test_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'loseCoverage')

# Display the selected columns
test_logReg_withReg.show(50)

# COMMAND ----------

train_logReg_withReg.write.saveAsTable("dua_058828_spa240.train_xg_boost_stage1", mode="overwrite")
test_logReg_withReg.write.saveAsTable("dua_058828_spa240.test_xg_boost_stage1", mode="overwrite")

# COMMAND ----------

# Get the metadata of the feature vector column
metadata = transformed_df.schema["features"].metadata
attrs = metadata["ml_attr"]["attrs"]

# Extract the one-hot encoded feature names from the metadata
one_hot_features = []
for attr in attrs.values():
    one_hot_features.extend([x["name"] for x in attr])

# Print the one-hot encoded feature names
print(one_hot_features)

# COMMAND ----------

# Get feature importances using the function (assuming 'features' is a list of feature names)   
features = ['ageCat_onehot_under10', 'ageCat_onehot_18To29', 'ageCat_onehot_10To17', 'ageCat_onehot_30To39', 'ageCat_onehot_50To64', 'ageCat_onehot_40To49', 'sex_onehot_female', 'sex_onehot_male', 'race_onehot_white', 'race_onehot_missing', 'race_onehot_black', 'race_onehot_hispanic', 'race_onehot_native', 'race_onehot_asian', 'race_onehot_hawaiian', 'state_onehot_IL', 'state_onehot_PA', 'state_onehot_MI', 'state_onehot_WA', 'state_onehot_AZ', 'state_onehot_TN', 'state_onehot_IN', 'state_onehot_MD', 'state_onehot_LA', 'state_onehot_KY', 'state_onehot_VA', 'state_onehot_AL', 'state_onehot_NM', 'state_onehot_NV', 'state_onehot_MS', 'state_onehot_WV', 'state_onehot_UT', 'state_onehot_KS', 'state_onehot_HI', 'state_onehot_ID', 'state_onehot_MT', 'state_onehot_ME', 'state_onehot_DC', 'state_onehot_DE', 'state_onehot_ND', 'state_onehot_VT', 'houseSize_onehot_missing', 'houseSize_onehot_twoToFive', 'houseSize_onehot_single', 'fedPovLine_onehot_missing', 'fedPovLine_onehot_0To100', 'fedPovLine_onehot_100To200', 'speakEnglish_onehot_missing', 'speakEnglish_onehot_yes', 'married_onehot_no', 'married_onehot_missing', 'UsCitizen_onehot_yes', 'UsCitizen_onehot_missing', 'ssi_onehot_no', 'ssi_onehot_missing', 'ssdi_onehot_no', 'ssdi_onehot_missing', 'tanf_onehot_no', 'tanf_onehot_missing', 'disabled_onehot_no', 'enrollMonth_onehot_Jan', 'enrollMonth_onehot_Feb', 'enrollMonth_onehot_Mar', 'enrollMonth_onehot_Oct', 'enrollMonth_onehot_Apr', 'enrollMonth_onehot_May', 'enrollMonth_onehot_Dec', 'enrollMonth_onehot_Nov', 'enrollMonth_onehot_Sep', 'enrollMonth_onehot_Aug', 'enrollMonth_onehot_Jul', 'enrollYear_onehot_2017', 'cov2016yes_onehot_1']

# Get feature importance
feature_importance = final_model.get_feature_importances(importance_type='gain')

# Print feature importance
for key, value in feature_importance.items():
    print(f"Feature: {key}, Importance: {value}")

# COMMAND ----------

# You can create a new dictionary like this:
named_feature_importance = {one_hot_features[int(key[1:])]: value for key, value in feature_importance.items()}

sorted_dict = dict(sorted(named_feature_importance.items(), key=lambda x: x[1], reverse=True))

# Then you can print the new dictionary like this:
for key, value in sorted_dict.items():
    print(f"{key}, {value}")