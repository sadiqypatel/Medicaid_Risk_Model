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
columns = ['ageCat','sex','race','state','enrollMonth','enrollYear','cov2016yes']

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

import mlflow
from pyspark.sql.functions import col
import math

if mlflow.active_run():
    mlflow.end_run()
    
# Set the active experiment
#mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/stage1/LogReg point estimate")
mlflow.set_tracking_uri("databricks")

# Perform an additional 80/20 train/validation split on the training set
train_val_ratio = 0.8
train_sub_df, val_df = train_df.randomSplit([train_val_ratio, 1 - train_val_ratio], seed=42)

# Now you have three DataFrames:
# - train_sub_df: The training subset used for training the model
# - val_df: The validation subset used for model evaluation and hyperparameter tuning
# - test_df: The test subset used for final model evaluation (previously created)

# Define the logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="loseCoverage")

# Define a list of hyperparameter combinations to try
hyperparameters = [
    {"regParam": 0.0, "elasticNetParam": 0.0}
    # Add more hyperparameter combinations as needed
]

# Initialize variables to keep track of the best model and its metrics
best_model = None
best_metrics = None
best_mcc = -1.0
best_run_id = None

# Define evaluators for AUC, accuracy, sensitivity, and PPV
auc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='loseCoverage', metricName='areaUnderROC')
multi_evaluator = MulticlassClassificationEvaluator(labelCol='loseCoverage', predictionCol='prediction')

# Check for an active MLflow run
active_run = mlflow.active_run()

if active_run is None:
    # If there is no active run, start a new one
    with mlflow.start_run(run_name="train_validate") as run:
        # Iterate through the hyperparameter combinations and train the model
    # If there is an active run, end it before starting a new one
        for params in hyperparameters:
            # Set hyperparameters for the current model
            lr.setRegParam(params["regParam"])
            lr.setElasticNetParam(params["elasticNetParam"])
            
            # Train the model and make predictions
            model = lr.fit(train_sub_df)
            predictions = model.transform(val_df)
            
            # Calculate metrics for the current model
            auc = auc_evaluator.evaluate(predictions)

            tp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 1.0)).count()
            tn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 0.0)).count()
            fp = predictions.filter((col('prediction') == 1.0) & (col('loseCoverage') == 0.0)).count()
            fn = predictions.filter((col('prediction') == 0.0) & (col('loseCoverage') == 1.0)).count()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
            specificity = tn / (tn + fp) if (tn + fp) > 0 else None
            ppv = tp / (tp + fp) if (tp + fp) > 0 else None
            npv = tn / (tn + fn) if (tn + fn) > 0 else None
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
            mcc = calculate_mcc(tn, fp, fn, tp)
                 
            # Log the hyperparameters and metrics for the current model to MLflow
            mlflow.log_params(params)
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
                best_run_id = run.info.run_id
                best_metrics = {
                    "AUC": auc,
                    "Accuracy": accuracy,
                    "Sensitivity": sensitivity,
                    "PPV": ppv,
                    "Specificity": specificity,
                    "MCC": mcc,
                    "NPV": npv
                }
    # Log the best hyperparameters to MLflow
    #mlflow.log_param("regParam", best_model._java_obj.getRegParam())
    #mlflow.log_param("elasticNetParam", best_model._java_obj.getElasticNetParam())

    # Log the metrics for the best model to MLflow
    mlflow.log_metrics(best_metrics)

    # Optionally, you can also log the best model itself to MLflow
    mlflow.spark.log_model(best_model, "best_model")
    mlflow.set_tag("model_description", "Best Logistic Regression model based on MCC")
    
    # Capture the current run ID
    run_id = mlflow.active_run().info.run_id
    print(run_id)
    run_info = mlflow.get_run(run_id)
    
    # Print the metric output for the best model
    for metric_name, metric_value in best_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

# COMMAND ----------

import mlflow
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

if mlflow.active_run():
    mlflow.end_run()
    
# Retrieve the run information from MLflow
run_info = mlflow.get_run(run_id)
        
# Extract the hyperparameters from the run information
best_hyperparams = run_info.data.params

# Create a new instance of the estimator with the same hyperparameters as the best_model
final_estimator = LogisticRegression(labelCol='loseCoverage', **best_hyperparams)

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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None
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

display(train_predictions)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT

# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])

# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
train_logReg = train_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
train_logReg = train_logReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'loseCoverage')

# Display the selected columns
train_logReg.show()

# COMMAND ----------

display(test_predictions)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT

# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])

# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
test_logReg = test_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
test_logReg = test_logReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'loseCoverage')

# Display the selected columns
test_logReg.show(50)

# COMMAND ----------

train_logReg.write.saveAsTable("dua_058828_spa240.train_log_reg_base_stage1", mode="overwrite")
test_logReg.write.saveAsTable("dua_058828_spa240.test_log_reg_base_stage1", mode="overwrite")

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
features = ['ageCat_onehot_under10', 'ageCat_onehot_18To29', 'ageCat_onehot_10To17', 'ageCat_onehot_30To39', 'ageCat_onehot_50To64', 'ageCat_onehot_40To49', 'sex_onehot_female', 'sex_onehot_male', 'race_onehot_white', 'race_onehot_missing', 'race_onehot_black', 'race_onehot_hispanic', 'race_onehot_asian', 'race_onehot_native', 'race_onehot_hawaiian', 'state_onehot_IL', 'state_onehot_PA', 'state_onehot_MI', 'state_onehot_AZ', 'state_onehot_WA', 'state_onehot_IN', 'state_onehot_LA', 'state_onehot_TN', 'state_onehot_MD', 'state_onehot_KY', 'state_onehot_VA', 'state_onehot_AL', 'state_onehot_NM', 'state_onehot_NV', 'state_onehot_MS', 'state_onehot_WV', 'state_onehot_KS', 'state_onehot_UT', 'state_onehot_HI', 'state_onehot_ID', 'state_onehot_MT', 'state_onehot_DE', 'state_onehot_DC', 'state_onehot_ME', 'state_onehot_VT', 'state_onehot_ND', 'enrollMonth_onehot_Jan', 'enrollMonth_onehot_Mar', 'enrollMonth_onehot_Feb', 'enrollMonth_onehot_Aug', 'enrollMonth_onehot_Nov', 'enrollMonth_onehot_Apr', 'enrollMonth_onehot_Dec', 'enrollMonth_onehot_May', 'enrollMonth_onehot_Jul', 'enrollMonth_onehot_Jun', 'enrollMonth_onehot_Oct', 'enrollYear_onehot_2017', 'cov2016yes_onehot_1']

importances = get_pyspark_logistic_regression_feature_importances(final_model, features)

# Display feature importances
for feature, importance in importances.items():
    print(f"{feature}: {importance}")

# COMMAND ----------

logged_model = 'dbfs:/databricks/mlflow/207340/0df84e96389e48a88bdbda40ddbeb5a3/artifacts/gbt-OOTB'

# Load model as a Spark UDF (for inferencing)
gbt_model_pyfunc = mlflow.pyfunc.spark_udf(spark,logged_model)

# Load model as a PySpark ML model (for developing)
gbt_model_spark = mlflow.spark.load_model(logged_model)