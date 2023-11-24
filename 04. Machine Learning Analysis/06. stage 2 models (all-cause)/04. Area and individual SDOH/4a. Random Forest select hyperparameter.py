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

from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_random_sample_5million_vector_assembler")
print(df.count())
df = df.withColumn("all_cause_binary", when(col("all_cause_acute_post") > 0, 1).otherwise(0))
df.groupBy("all_cause_binary").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = 50000 / df.count()

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
mlflow.set_experiment("/Users/SPA240/Paper 1/analysis/6. stage 2 models (all-cause)/4a. Random Forest select hyperparameter")
mlflow.set_tracking_uri("databricks")

# Create an instance of the XGBoostClassifier
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="all_cause_binary")

#Create an instance of the ParamGridBuilder and define the parameter grid
#Define a more comprehensive hyperparameter grid

# grid search based on model
param_grid = (
    ParamGridBuilder()
    .addGrid(rf_classifier.minInstancesPerNode, [1, 10, 20, 30, 40, 50, 60])                 
    .addGrid(rf_classifier.featureSubsetStrategy, ["auto","sqrt","log2"])                       
    .addGrid(rf_classifier.maxDepth, [10, 12, 14, 16, 18, 20])                         
    .addGrid(rf_classifier.numTrees, [10, 20, 50, 75, 100, 125, 150, 175, 200])                                
    .build()
)

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

# Loop through each hyperparameter combination in the param_grid
for params in param_grid:
    rf = RandomForestClassifier(featuresCol="features", labelCol="all_cause_binary", **{param.name: value for param, value in params.items()})
    model = rf.fit(train_sub_df)
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

    # Log the hyperparameters and metrics to MLflow
    with mlflow.start_run():
        # Log hyperparameters  
        mlflow.log_param("min_samples_leaf", params[rf_classifier.minInstancesPerNode])
        mlflow.log_param("max_features",  params[rf_classifier.featureSubsetStrategy])
        mlflow.log_param("max_depth", params[rf_classifier.maxDepth])
        mlflow.log_param("numb_of_estim", params[rf_classifier.numTrees])
              
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