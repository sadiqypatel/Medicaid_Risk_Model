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
import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# COMMAND ----------

df = spark.table("dua_058828_spa240.paper1_stage2_final_data_kids_2M")

print("Features:")
for feature in df.columns:
    print(feature)

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction =    df.count() / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)
train_data, test_data = sampled_df.randomSplit([0.8, 0.2], seed=123)
train_val_ratio = 0.8
train_sub_df, val_df = train_data.randomSplit([train_val_ratio, 1 - train_val_ratio], seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/cost new hyper/3a. xgboost kids all features")
mlflow.set_tracking_uri("databricks")

# COMMAND ----------

xgb_classifier = XgboostRegressor(labelCol="total_cost", featuresCol="features_all_sdoh", missing=0.0)

param_grid = (
    ParamGridBuilder()
    .addGrid(xgb_classifier.learning_rate, [0.01])                 
    .addGrid(xgb_classifier.subsample, [0.4])                       
    .addGrid(xgb_classifier.min_child_weight, [0])                         
    .addGrid(xgb_classifier.colsample_bytree, [0.5])                   
    .addGrid(xgb_classifier.max_depth, [18, 20, 22])                        
    .addGrid(xgb_classifier.reg_alpha, [0.0, 0.5])                           
    .addGrid(xgb_classifier.reg_lambda, [0.0, 0.5])                          
    .addGrid(xgb_classifier.gamma, [0.0, 0.5, 1.0])                                
    .addGrid(xgb_classifier.colsample_bylevel, [0.5, 0.75, 1.0])                   
    .build()
)

# COMMAND ----------

best_model = None
best_metrics = None
best_r2 = -100
best_run_id = None

if mlflow.active_run():
    mlflow.end_run()

# Iterate over the parameter grid
for params in param_grid:
    # Log the current set of parameters
    #param_dict = {param.name: value for param, value in params.items()}
    #mlflow.log_params(params)
    # Train the XGBoost model with the current set of parameters
    param_dict = {param.name: value for param, value in params.items()}
    xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_baseline", missing=0.0)
    xgb_model.setParams(**param_dict)  # Set the model parameters
    xgb_model = xgb_model.fit(train_sub_df)
    # Make predictions and evaluate the model
    predictions = xgb_model.transform(val_df)
    evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
    r2 = evaluator.evaluate(predictions)
    print(r2)
    
#     learning_rate = params['learning_rate']
#     subsample = params['subsample']
#     min_child_weight = params['min_child_weight']
#     colsample_bytree = params['colsample_bytree']
#     max_depth = params['max_depth']
#     reg_alpha = params['reg_alpha']
#     reg_lambda = params['reg_lambda']
#     gamma = params['gamma']
#     colsample_bylevel = params['colsample_bylevel']
    
    with mlflow.start_run():
        mlflow.log_params(param_dict)
        mlflow.log_metric("R-squared", r2)        
        
        if r2 > best_r2:
            best_model = xgb_model
            best_r2 = r2
            best_run_id = mlflow.active_run().info.run_id
            best_run_info = mlflow.get_run(best_run_id)

#             best_params = {
#             "best_learning_rate": learning_rate,
#             "best_subsample": subsample,
#             "best_min_child_weight": min_child_weight,
#             "best_colsample_bytree": colsample_bytree,
#             "best_max_depth": max_depth,
#             "best_reg_alpha": reg_alpha,
#             "best_reg_lambda": reg_lambda,
#             "best_gamma": gamma,
#             "best_colsample_bylevel": colsample_bylevel
#             }

# # Capture the current run ID
run_id = best_run_id
run_info = best_run_info
#print(run_info)
params = best_run_info.data.params
#print(params)

if mlflow.active_run():
    mlflow.end_run()
    
# Retrieve the run information from MLflow
run_info = mlflow.get_run(run_id)
        
# # Extract the hyperparameters from the run information
params = run_info.data.params
params = {k: float(v) for k, v in params.items()}
params['max_depth'] = int(float(params['max_depth']))
print(params)

# Create a new instance of the estimator with the same hyperparameters as the best_model
final_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_no_sdoh", missing=0.0)
final_model.setParams(**params)  # Set the model parameters
final_model = final_model.fit(train_data)
test_predictions = final_model.transform(test_data)
train_predictions = final_model.transform(train_data)
evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
r2_test = evaluator.evaluate(predictions)
r2_train = evaluator.evaluate(train_predictions)
print("r2 for test data")
print(r2_test)
print("r2 for train data")
print(r2_train)

test_predictions.write.saveAsTable("dua_058828_spa240.test_cost_all_features_kids", mode="overwrite")
display(test_predictions)