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

df = spark.table("dua_058828_spa240.paper1_stage2_final_data")

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
fraction = 375000 / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

# Continue with your analysis using the sampled DataFrame 'sampled_df'

# COMMAND ----------

xgb_classifier = XgboostRegressor(labelCol="total_cost", featuresCol="features_all_sdoh", missing=0.0)

param_grid = (
    ParamGridBuilder()
    .addGrid(xgb_classifier.learning_rate, [0.01])                 
    .addGrid(xgb_classifier.subsample, [0.4, 0.55, 0.7])                       
    .addGrid(xgb_classifier.min_child_weight, [0, 2.5, 5])                         
    .addGrid(xgb_classifier.colsample_bytree, [0.5, 0.75, 1.0])                   
    .addGrid(xgb_classifier.max_depth, [16, 19, 22])                        
    .addGrid(xgb_classifier.reg_alpha, [0.0, 0.5, 1.0])                           
    .addGrid(xgb_classifier.reg_lambda, [0.0, 0.5, 1.0])                          
    .addGrid(xgb_classifier.gamma, [0.0, 0.5, 1.0])                                
    .addGrid(xgb_classifier.colsample_bylevel, [0.5, 0.75, 1.0])                   
    .build()
)

# COMMAND ----------

# Select the relevant columns
sampled_df = sampled_df.select("features_area_sdoh", "total_cost")

# COMMAND ----------

# # # Perform an additional 80/20 train/validation split on the training set
train_data, test_data = sampled_df.randomSplit([0.8, 0.2], seed=123)
train_val_ratio = 0.8
train_sub_df, val_df = train_data.randomSplit([train_val_ratio, 1 - train_val_ratio], seed=42)

# COMMAND ----------

# # # # Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/cost analysis/9d. xgboost area SDOH")
mlflow.set_tracking_uri("databricks")

# Iterate over the parameter grid
for params in param_grid:
    # Log the current set of parameters
    param_dict = {param.name: value for param, value in params.items()}
    #mlflow.log_params(params)

    # Train the XGBoost model with the current set of parameters
    xgb_model = XgboostRegressor(labelCol="total_cost", featuresCol="features_area_sdoh", missing=0.0)
    xgb_model.setParams(**param_dict)  # Set the model parameters

    xgb_model = xgb_model.fit(train_sub_df)

    # Make predictions and evaluate the model
    predictions = xgb_model.transform(val_df)
    evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    # Log the evaluation metric
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run():
        mlflow.log_params(param_dict)
        mlflow.log_metric("R-squared", r2)

# COMMAND ----------

