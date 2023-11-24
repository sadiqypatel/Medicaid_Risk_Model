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
from sparkdl.xgboost import XgboostRegressor
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

# COMMAND ----------

df = spark.table("dua_058828_spa240.paper1_stage2_final_data_adults")

# # Convert 'total_cost' column to VectorUDT
# vectorize_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
# df = df.withColumn("total_cost", vectorize_udf(df["total_cost"]))

# Print all the features
print("Features:")
for feature in df.columns:
    print(feature)

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = df.count() / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

# Continue with your analysis using the sampled DataFrame 'sampled_df'

# COMMAND ----------

# # # # Set the active experiment
#mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/cost analysis/9a. xgboost no sdoh")
# mlflow.set_tracking_uri("databricks")

# COMMAND ----------

# Select the relevant columns
sampled_df1 = sampled_df.select("beneID", "state","features_baseline", "total_cost")

# # # Perform an additional 80/20 train/validation split on the training set
train_data, test_data = sampled_df1.randomSplit([0.8, 0.2], seed=123)

params = 
{
'colsample_bylevel': 0.75, 'colsample_bytree': 0.5, 'gamma': 0,   'learning_rate': 0.01, 'max_depth': 22, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4
}

xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_baseline", missing=0.0, **params)
xgb_model.setParams(**params)  # Set the model parameters
xgb_model = xgb_model.fit(train_data)
# Make predictions and evaluate the model
predictions = xgb_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(r2)

# COMMAND ----------

# Select the relevant columns
sampled_df1 = sampled_df.select("beneID", "state","features_no_sdoh", "total_cost")

# # # Perform an additional 80/20 train/validation split on the training set
train_data, test_data = sampled_df1.randomSplit([0.8, 0.2], seed=123)

best_hyperparams = [
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 1,   'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.75,  'colsample_bytree': 0.5, 'gamma': 1,   'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.75,  'colsample_bytree': 0.5, 'gamma': 0,   'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 1,   'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0,   'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 1.0, 'reg_lambda': 0.0, 'subsample': 0.4}     
]

# Iterate over the parameter grid
for params in best_hyperparams:
    # Log the current set of parameters
    #param_dict = {param.name: value for param, value in params.items()}
    #mlflow.log_params(params)
    # Train the XGBoost model with the current set of parameters
    xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_no_sdoh", missing=0.0, **params)
    xgb_model.setParams(**params)  # Set the model parameters
    xgb_model = xgb_model.fit(train_data)
    # Make predictions and evaluate the model
    predictions = xgb_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print(r2)
#     # Log the evaluation metric
#     if mlflow.active_run():
#         mlflow.end_run()
#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metric("R-squared", r2)

# COMMAND ----------

# Select the relevant columns
sampled_df1 = sampled_df.select("beneID", "state", "features_area_sdoh", "total_cost")

# # # Perform an additional 80/20 train/validation split on the training set
train_data, test_data = sampled_df1.randomSplit([0.8, 0.2], seed=123)

best_hyperparams = [
     {'colsample_bylevel': 1.0,  'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 1, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0,   'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.75,   'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,  'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,  'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 1.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.5, 'subsample': 0.4}   
]

# Iterate over the parameter grid
for params in best_hyperparams:
    # Log the current set of parameters
    #param_dict = {param.name: value for param, value in params.items()}
    #mlflow.log_params(params)
    # Train the XGBoost model with the current set of parameters
    xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_area_sdoh", missing=0.0, **params)
    xgb_model.setParams(**params)  # Set the model parameters
    xgb_model = xgb_model.fit(train_data)
    # Make predictions and evaluate the model
    predictions = xgb_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print(r2)
    # Log the evaluation metric
#     if mlflow.active_run():
#         mlflow.end_run()
#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metric("R-squared", r2)

# COMMAND ----------

# Select the relevant columns
sampled_df1 = sampled_df.select("beneID", "state", "features_all_sdoh", "total_cost")

# # # Perform an additional 80/20 train/validation split on the training set
train_data, test_data = sampled_df1.randomSplit([0.8, 0.2], seed=123)

best_hyperparams = [
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 1.0, 'reg_lambda': 0.5, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 16, 'min_child_weight': 0, 'reg_alpha': 1.0, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 0.5, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.0, 'subsample': 0.4},
     {'colsample_bylevel': 0.5,   'colsample_bytree': 0.5, 'gamma': 1.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.0, 'reg_lambda': 0.5, 'subsample': 0.4},
     {'colsample_bylevel': 1.0,   'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.01, 'max_depth': 19, 'min_child_weight': 0, 'reg_alpha': 0.5, 'reg_lambda': 0.5, 'subsample': 0.4}     
]

# Iterate over the parameter grid
for params in best_hyperparams:
    # Log the current set of parameters
    #param_dict = {param.name: value for param, value in params.items()}
    #mlflow.log_params(params)
    # Train the XGBoost model with the current set of parameters
    xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_all_sdoh", missing=0.0, **params)
    xgb_model.setParams(**params)  # Set the model parameters
    xgb_model = xgb_model.fit(train_data)
    # Make predictions and evaluate the model
    predictions = xgb_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print(r2)
    # Log the evaluation metric
#     if mlflow.active_run():
#         mlflow.end_run()
#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metric("R-squared", r2)