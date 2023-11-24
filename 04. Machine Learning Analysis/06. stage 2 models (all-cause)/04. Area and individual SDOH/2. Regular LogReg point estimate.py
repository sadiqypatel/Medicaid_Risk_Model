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

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_random_sample_5million_vector_assembler")
print(df.count())
df = df.withColumn("all_cause_binary", when(col("all_cause_acute_post") > 0, 1).otherwise(0))
df.groupBy("all_cause_binary").count().orderBy(col("count").desc()).show()

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = df.count() / df.count()

#fraction = 10000000 / df.count()

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

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from pyspark.sql.functions import col
import math

if mlflow.active_run():
    mlflow.end_run()

# Set the active experiment
mlflow.set_experiment("/Users/SPA240/riskScoreCode/analysis/6. stage 2 models (all-cause)/2a. LogReg with Regular point estimate")
mlflow.set_tracking_uri("databricks")

# Perform an additional 80/20 train/validation split on the training set
train_val_ratio = 0.8
train_sub_df, val_df = train_df.randomSplit([train_val_ratio, 1 - train_val_ratio], seed=42)

# Create an instance of the LogisticRegression model
lr = LogisticRegression(featuresCol="features", labelCol="all_cause_binary")

#Create an instance of the ParamGridBuilder and define the parameter grid
param_grid = (ParamGridBuilder()
              .addGrid(lr.regParam, [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
              .addGrid(lr.elasticNetParam, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
              .build())

# param_grid = (ParamGridBuilder()
#               .addGrid(lr.regParam, [0.001, 0.01, 100])
#               .addGrid(lr.elasticNetParam, [0.2, 1.0])
#               .build())

# Initialize variables to keep track of the best model and its metrics
best_model = None
best_metrics = None
best_mcc = -1.0
best_run_id = None

# Define evaluators for AUC, accuracy, sensitivity, and PPV
auc_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='all_cause_binary', metricName='areaUnderROC')
multi_evaluator = MulticlassClassificationEvaluator(labelCol='all_cause_binary', predictionCol='prediction')

for params in param_grid:
    regParam_value = params[lr.regParam]
    elasticNetParam_value = params[lr.elasticNetParam] 
    with mlflow.start_run(run_name="train_validate") as run:
        lr.setRegParam(regParam_value).setElasticNetParam(elasticNetParam_value)
        model = lr.fit(train_sub_df)
        predictions = model.transform(val_df)

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

        mlflow.log_param("elasticNetParam", elasticNetParam_value)
        mlflow.log_param("regParam", regParam_value)

        # Log the hyperparameters and metrics for the current model to MLflow
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
            best_elastic = mlflow.log_param("elasticNetParam", elasticNetParam_value)
            best_regparam = mlflow.log_param("regParam", regParam_value)
            best_metrics = {
            "AUC": auc,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "PPV": ppv,
            "Specificity": specificity,
            "MCC": mcc,
            "NPV": npv
                }
            
            # Optionally, you can also log the best model itself to MLflow
            mlflow.spark.log_model(best_model, "best_model")
            mlflow.set_tag("model_description", "Best Log Reg w/ regularization based on MCC")
            mlflow.log_metrics(best_metrics)
            
# Log the metrics for the best model to MLflow
#mlflow.log_metrics(best_metrics)
#mlflow.log_param("best_elastic_param", best_elastic)
#mlflow.log_param("best_reg_param", best_regparam)

# Capture the current run ID
run_id = best_run_id
run_info = best_run_info

# Print the metric output for the best model
for metric_name, metric_value in best_metrics.items():
    print(f"{metric_name}: {metric_value:.5f}")

# COMMAND ----------

print(run_info)
print(best_run_info)

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
        
# Extract the hyperparameters from the run information
best_hyperparams = run_info.data.params
print(best_hyperparams)

# Convert string values to floats using a dictionary comprehension
best_hyperparams = {k: float(v) for k, v in best_hyperparams.items()}
print(best_hyperparams)

# Create a new instance of the estimator with the same hyperparameters as the best_model
final_estimator = LogisticRegression(featuresCol="features", labelCol='all_cause_binary', **best_hyperparams)

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
train_logReg_withReg = train_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
train_logReg_withReg = train_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'all_cause_binary')

# Display the selected columns
train_logReg_withReg.show()

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
test_logReg_withReg = test_predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
test_logReg_withReg = test_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'all_cause_binary')

# Display the selected columns
test_logReg_withReg.show(50)

# COMMAND ----------

train_logReg_withReg.write.saveAsTable("dua_058828_spa240.stage2_train_all_cause_logReg_withReg", mode="overwrite")
test_logReg_withReg.write.saveAsTable("dua_058828_spa240.stage2_test_all_cause_logReg_withReg", mode="overwrite")

# COMMAND ----------

# Get the metadata of the feature vector column
metadata = sampled_df.schema["features"].metadata
attrs = metadata["ml_attr"]["attrs"]

# Extract the one-hot encoded feature names from the metadata
one_hot_features = []
for attr in attrs.values():
    one_hot_features.extend([x["name"] for x in attr])

# Print the one-hot encoded feature names
print(one_hot_features)

# COMMAND ----------

# Get feature importances using the function (assuming 'features' is a list of feature names)   
features = ['numeric_features_stage1_lose_coverage_prob', 'numeric_features_saServRate', 'numeric_features_saFacRate', 'numeric_features_mhTreatRate', 'numeric_features_popDensity', 'numeric_features_povRate', 'numeric_features_publicAssistRate', 'numeric_features_highSchoolGradRate', 'numeric_features_goodAirDays', 'numeric_features_injDeathRate', 'numeric_features_urgentCareRate', 'numeric_features_drugdeathRate', 'numeric_features_100HeatDays', 'numeric_features_aprnRate', 'numeric_features_allcause_slope', 'numeric_features_avoid_slope', 'numeric_features_total_inpatient_days', 'numeric_features_total_all_cause_visits', 'numeric_features_percent_non_emergent', 'numeric_features_pharm_slope', 'numeric_features_number_of_fills', 'numeric_features_number_of_unique_med', 'numeric_features_percent_generic', 'numeric_features_percent_med_adherence', 'numeric_features_long_term_episodes', 'numeric_features_long_term_days', 'numeric_features_ccsr_null', 'numeric_features_BLD001', 'numeric_features_BLD002', 'numeric_features_BLD003', 'numeric_features_BLD004', 'numeric_features_BLD005', 'numeric_features_BLD006', 'numeric_features_BLD007', 'numeric_features_BLD008', 'numeric_features_BLD009', 'numeric_features_BLD010', 'numeric_features_CIR001', 'numeric_features_CIR002', 'numeric_features_CIR003', 'numeric_features_CIR004', 'numeric_features_CIR005', 'numeric_features_CIR006', 'numeric_features_CIR007', 'numeric_features_CIR008', 'numeric_features_CIR009', 'numeric_features_CIR010', 'numeric_features_CIR011', 'numeric_features_CIR012', 'numeric_features_CIR013', 'numeric_features_CIR014', 'numeric_features_CIR015', 'numeric_features_CIR016', 'numeric_features_CIR017', 'numeric_features_CIR018', 'numeric_features_CIR019', 'numeric_features_CIR020', 'numeric_features_CIR021', 'numeric_features_CIR022', 'numeric_features_CIR023', 'numeric_features_CIR024', 'numeric_features_CIR025', 'numeric_features_CIR026', 'numeric_features_CIR027', 'numeric_features_CIR028', 'numeric_features_CIR029', 'numeric_features_CIR030', 'numeric_features_CIR031', 'numeric_features_CIR032', 'numeric_features_CIR033', 'numeric_features_CIR034', 'numeric_features_CIR035', 'numeric_features_CIR036', 'numeric_features_CIR037', 'numeric_features_CIR038', 'numeric_features_CIR039', 'numeric_features_DEN001', 'numeric_features_DIG001', 'numeric_features_DIG004', 'numeric_features_DIG005', 'numeric_features_DIG006', 'numeric_features_DIG007', 'numeric_features_DIG008', 'numeric_features_DIG009', 'numeric_features_DIG010', 'numeric_features_DIG011', 'numeric_features_DIG012', 'numeric_features_DIG013', 'numeric_features_DIG014', 'numeric_features_DIG015', 'numeric_features_DIG016', 'numeric_features_DIG017', 'numeric_features_DIG018', 'numeric_features_DIG019', 'numeric_features_DIG020', 'numeric_features_DIG021', 'numeric_features_DIG022', 'numeric_features_DIG023', 'numeric_features_DIG024', 'numeric_features_DIG025', 'numeric_features_EAR001', 'numeric_features_EAR002', 'numeric_features_EAR003', 'numeric_features_EAR004', 'numeric_features_EAR005', 'numeric_features_EAR006', 'numeric_features_END001', 'numeric_features_END002', 'numeric_features_END003', 'numeric_features_END007', 'numeric_features_END008', 'numeric_features_END009', 'numeric_features_END010', 'numeric_features_END011', 'numeric_features_END012', 'numeric_features_END013', 'numeric_features_END014', 'numeric_features_END015', 'numeric_features_END016', 'numeric_features_END017', 'numeric_features_EXT001', 'numeric_features_EXT002', 'numeric_features_EXT003', 'numeric_features_EXT004', 'numeric_features_EXT005', 'numeric_features_EXT006', 'numeric_features_EXT007', 'numeric_features_EXT008', 'numeric_features_EXT009', 'numeric_features_EXT010', 'numeric_features_EXT011', 'numeric_features_EXT012', 'numeric_features_EXT013', 'numeric_features_EXT014', 'numeric_features_EXT015', 'numeric_features_EXT016', 'numeric_features_EXT017', 'numeric_features_EXT018', 'numeric_features_EXT019', 'numeric_features_EXT025', 'numeric_features_EXT026', 'numeric_features_EXT027', 'numeric_features_EXT028', 'numeric_features_EXT029', 'numeric_features_EXT030', 'numeric_features_EYE001', 'numeric_features_EYE002', 'numeric_features_EYE003', 'numeric_features_EYE004', 'numeric_features_EYE005', 'numeric_features_EYE006', 'numeric_features_EYE007', 'numeric_features_EYE008', 'numeric_features_EYE009', 'numeric_features_EYE010', 'numeric_features_EYE011', 'numeric_features_EYE012', 'numeric_features_FAC001', 'numeric_features_FAC002', 'numeric_features_FAC003', 'numeric_features_FAC004', 'numeric_features_FAC005', 'numeric_features_FAC006', 'numeric_features_FAC007', 'numeric_features_FAC008', 'numeric_features_FAC009', 'numeric_features_FAC010', 'numeric_features_FAC011', 'numeric_features_FAC012', 'numeric_features_FAC013', 'numeric_features_FAC014', 'numeric_features_FAC015', 'numeric_features_FAC016', 'numeric_features_FAC017', 'numeric_features_FAC018', 'numeric_features_FAC019', 'numeric_features_FAC020', 'numeric_features_FAC021', 'numeric_features_FAC022', 'numeric_features_FAC023', 'numeric_features_FAC024', 'numeric_features_FAC025', 'numeric_features_GEN001', 'numeric_features_GEN002', 'numeric_features_GEN003', 'numeric_features_GEN004', 'numeric_features_GEN005', 'numeric_features_GEN006', 'numeric_features_GEN007', 'numeric_features_GEN008', 'numeric_features_GEN009', 'numeric_features_GEN010', 'numeric_features_GEN011', 'numeric_features_GEN012', 'numeric_features_GEN013', 'numeric_features_GEN014', 'numeric_features_GEN015', 'numeric_features_GEN016', 'numeric_features_GEN017', 'numeric_features_GEN018', 'numeric_features_GEN019', 'numeric_features_GEN020', 'numeric_features_GEN021', 'numeric_features_GEN022', 'numeric_features_GEN023', 'numeric_features_GEN024', 'numeric_features_GEN025', 'numeric_features_GEN026', 'numeric_features_INF001', 'numeric_features_INF002', 'numeric_features_INF003', 'numeric_features_INF004', 'numeric_features_INF005', 'numeric_features_INF006', 'numeric_features_INF007', 'numeric_features_INF008', 'numeric_features_INF009', 'numeric_features_INF010', 'numeric_features_INF011', 'numeric_features_INJ001', 'numeric_features_INJ002', 'numeric_features_INJ003', 'numeric_features_INJ004', 'numeric_features_INJ005', 'numeric_features_INJ006', 'numeric_features_INJ007', 'numeric_features_INJ008', 'numeric_features_INJ009', 'numeric_features_INJ010', 'numeric_features_INJ011', 'numeric_features_INJ012', 'numeric_features_INJ013', 'numeric_features_INJ014', 'numeric_features_INJ015', 'numeric_features_INJ016', 'numeric_features_INJ017', 'numeric_features_INJ018', 'numeric_features_INJ019', 'numeric_features_INJ021', 'numeric_features_INJ024', 'numeric_features_INJ025', 'numeric_features_INJ026', 'numeric_features_INJ027', 'numeric_features_INJ028', 'numeric_features_INJ029', 'numeric_features_INJ030', 'numeric_features_INJ031', 'numeric_features_INJ032', 'numeric_features_INJ033', 'numeric_features_INJ034', 'numeric_features_INJ035', 'numeric_features_INJ036', 'numeric_features_INJ037', 'numeric_features_INJ038', 'numeric_features_INJ039', 'numeric_features_INJ040', 'numeric_features_INJ041', 'numeric_features_INJ042', 'numeric_features_INJ043', 'numeric_features_INJ044', 'numeric_features_INJ045', 'numeric_features_INJ046', 'numeric_features_INJ047', 'numeric_features_INJ048', 'numeric_features_INJ049', 'numeric_features_INJ050', 'numeric_features_INJ051', 'numeric_features_INJ052', 'numeric_features_INJ053', 'numeric_features_INJ054', 'numeric_features_INJ055', 'numeric_features_INJ056', 'numeric_features_INJ057', 'numeric_features_INJ058', 'numeric_features_INJ059', 'numeric_features_INJ060', 'numeric_features_INJ061', 'numeric_features_INJ062', 'numeric_features_INJ063', 'numeric_features_INJ064', 'numeric_features_INJ065', 'numeric_features_INJ066', 'numeric_features_INJ067', 'numeric_features_INJ068', 'numeric_features_INJ069', 'numeric_features_INJ070', 'numeric_features_INJ071', 'numeric_features_INJ072', 'numeric_features_INJ073', 'numeric_features_INJ074', 'numeric_features_INJ075', 'numeric_features_INJ076', 'numeric_features_MAL001', 'numeric_features_MAL002', 'numeric_features_MAL003', 'numeric_features_MAL004', 'numeric_features_MAL005', 'numeric_features_MAL006', 'numeric_features_MAL007', 'numeric_features_MAL008', 'numeric_features_MAL009', 'numeric_features_MAL010', 'numeric_features_MBD001', 'numeric_features_MBD002', 'numeric_features_MBD003', 'numeric_features_MBD004', 'numeric_features_MBD005', 'numeric_features_MBD006', 'numeric_features_MBD007', 'numeric_features_MBD008', 'numeric_features_MBD009', 'numeric_features_MBD010', 'numeric_features_MBD011', 'numeric_features_MBD012', 'numeric_features_MBD013', 'numeric_features_MBD014', 'numeric_features_MBD017', 'numeric_features_MBD018', 'numeric_features_MBD019', 'numeric_features_MBD020', 'numeric_features_MBD021', 'numeric_features_MBD022', 'numeric_features_MBD023', 'numeric_features_MBD024', 'numeric_features_MBD025', 'numeric_features_MBD026', 'numeric_features_MUS001', 'numeric_features_MUS002', 'numeric_features_MUS003', 'numeric_features_MUS004', 'numeric_features_MUS005', 'numeric_features_MUS006', 'numeric_features_MUS007', 'numeric_features_MUS008', 'numeric_features_MUS009', 'numeric_features_MUS010', 'numeric_features_MUS011', 'numeric_features_MUS012', 'numeric_features_MUS013', 'numeric_features_MUS014', 'numeric_features_MUS015', 'numeric_features_MUS016', 'numeric_features_MUS017', 'numeric_features_MUS018', 'numeric_features_MUS019', 'numeric_features_MUS020', 'numeric_features_MUS021', 'numeric_features_MUS022', 'numeric_features_MUS023', 'numeric_features_MUS024', 'numeric_features_MUS025', 'numeric_features_MUS026', 'numeric_features_MUS028', 'numeric_features_MUS030', 'numeric_features_MUS031', 'numeric_features_MUS032', 'numeric_features_MUS033', 'numeric_features_MUS034', 'numeric_features_MUS036', 'numeric_features_MUS037', 'numeric_features_MUS038', 'numeric_features_NEO001', 'numeric_features_NEO002', 'numeric_features_NEO003', 'numeric_features_NEO004', 'numeric_features_NEO005', 'numeric_features_NEO006', 'numeric_features_NEO007', 'numeric_features_NEO008', 'numeric_features_NEO009', 'numeric_features_NEO010', 'numeric_features_NEO011', 'numeric_features_NEO012', 'numeric_features_NEO013', 'numeric_features_NEO014', 'numeric_features_NEO015', 'numeric_features_NEO016', 'numeric_features_NEO017', 'numeric_features_NEO018', 'numeric_features_NEO019', 'numeric_features_NEO020', 'numeric_features_NEO021', 'numeric_features_NEO022', 'numeric_features_NEO023', 'numeric_features_NEO024', 'numeric_features_NEO025', 'numeric_features_NEO026', 'numeric_features_NEO027', 'numeric_features_NEO028', 'numeric_features_NEO029', 'numeric_features_NEO030', 'numeric_features_NEO031', 'numeric_features_NEO032', 'numeric_features_NEO033', 'numeric_features_NEO034', 'numeric_features_NEO035', 'numeric_features_NEO036', 'numeric_features_NEO037', 'numeric_features_NEO038', 'numeric_features_NEO039', 'numeric_features_NEO040', 'numeric_features_NEO041', 'numeric_features_NEO042', 'numeric_features_NEO043', 'numeric_features_NEO044', 'numeric_features_NEO045', 'numeric_features_NEO046', 'numeric_features_NEO047', 'numeric_features_NEO048', 'numeric_features_NEO049', 'numeric_features_NEO050', 'numeric_features_NEO051', 'numeric_features_NEO052', 'numeric_features_NEO053', 'numeric_features_NEO054', 'numeric_features_NEO055', 'numeric_features_NEO056', 'numeric_features_NEO057', 'numeric_features_NEO058', 'numeric_features_NEO059', 'numeric_features_NEO060', 'numeric_features_NEO061', 'numeric_features_NEO062', 'numeric_features_NEO063', 'numeric_features_NEO064', 'numeric_features_NEO065', 'numeric_features_NEO066', 'numeric_features_NEO067', 'numeric_features_NEO068', 'numeric_features_NEO069', 'numeric_features_NEO070', 'numeric_features_NEO071', 'numeric_features_NEO072', 'numeric_features_NEO073', 'numeric_features_NEO074', 'numeric_features_NVS001', 'numeric_features_NVS002', 'numeric_features_NVS003', 'numeric_features_NVS004', 'numeric_features_NVS005', 'numeric_features_NVS006', 'numeric_features_NVS007', 'numeric_features_NVS008', 'numeric_features_NVS009', 'numeric_features_NVS010', 'numeric_features_NVS011', 'numeric_features_NVS012', 'numeric_features_NVS013', 'numeric_features_NVS014', 'numeric_features_NVS015', 'numeric_features_NVS016', 'numeric_features_NVS017', 'numeric_features_NVS018', 'numeric_features_NVS019', 'numeric_features_NVS020', 'numeric_features_NVS021', 'numeric_features_NVS022', 'numeric_features_PNL001', 'numeric_features_PNL002', 'numeric_features_PNL003', 'numeric_features_PNL004', 'numeric_features_PNL005', 'numeric_features_PNL006', 'numeric_features_PNL007', 'numeric_features_PNL008', 'numeric_features_PNL009', 'numeric_features_PNL010', 'numeric_features_PNL011', 'numeric_features_PNL012', 'numeric_features_PNL013', 'numeric_features_PNL014', 'numeric_features_PRG001', 'numeric_features_PRG002', 'numeric_features_PRG003', 'numeric_features_PRG004', 'numeric_features_PRG005', 'numeric_features_PRG006', 'numeric_features_PRG007', 'numeric_features_PRG008', 'numeric_features_PRG009', 'numeric_features_PRG010', 'numeric_features_PRG011', 'numeric_features_PRG012', 'numeric_features_PRG013', 'numeric_features_PRG014', 'numeric_features_PRG015', 'numeric_features_PRG016', 'numeric_features_PRG017', 'numeric_features_PRG018', 'numeric_features_PRG020', 'numeric_features_PRG021', 'numeric_features_PRG022', 'numeric_features_PRG023', 'numeric_features_PRG024', 'numeric_features_PRG025', 'numeric_features_PRG026', 'numeric_features_PRG027', 'numeric_features_PRG028', 'numeric_features_PRG029', 'numeric_features_PRG030', 'numeric_features_RSP001', 'numeric_features_RSP002', 'numeric_features_RSP003', 'numeric_features_RSP004', 'numeric_features_RSP005', 'numeric_features_RSP006', 'numeric_features_RSP007', 'numeric_features_RSP008', 'numeric_features_RSP009', 'numeric_features_RSP010', 'numeric_features_RSP011', 'numeric_features_RSP012', 'numeric_features_RSP013', 'numeric_features_RSP014', 'numeric_features_RSP015', 'numeric_features_RSP016', 'numeric_features_RSP017', 'numeric_features_SKN001', 'numeric_features_SKN002', 'numeric_features_SKN003', 'numeric_features_SKN004', 'numeric_features_SKN005', 'numeric_features_SKN006', 'numeric_features_SKN007', 'numeric_features_SYM001', 'numeric_features_SYM002', 'numeric_features_SYM003', 'numeric_features_SYM004', 'numeric_features_SYM005', 'numeric_features_SYM006', 'numeric_features_SYM007', 'numeric_features_SYM008', 'numeric_features_SYM009', 'numeric_features_SYM010', 'numeric_features_SYM011', 'numeric_features_SYM012', 'numeric_features_SYM013', 'numeric_features_SYM014', 'numeric_features_SYM015', 'numeric_features_SYM016', 'numeric_features_SYM017', 'numeric_features_betos_null', 'numeric_features_AA000N', 'numeric_features_DA000N', 'numeric_features_DA018N', 'numeric_features_DB000N', 'numeric_features_DC000N', 'numeric_features_DC002N', 'numeric_features_DD000N', 'numeric_features_DD009N', 'numeric_features_DD021N', 'numeric_features_DE000N', 'numeric_features_DE001N', 'numeric_features_DE005N', 'numeric_features_DE012N', 'numeric_features_DE013N', 'numeric_features_DE014N', 'numeric_features_DE015N', 'numeric_features_DE016N', 'numeric_features_DE017N', 'numeric_features_DE020N', 'numeric_features_DE022N', 'numeric_features_DF000N', 'numeric_features_DF003N', 'numeric_features_DF007N', 'numeric_features_DF008N', 'numeric_features_DF010N', 'numeric_features_DF011N', 'numeric_features_DF019N', 'numeric_features_DG000N', 'numeric_features_DG004N', 'numeric_features_DG006N', 'numeric_features_EB000N', 'numeric_features_EB009N', 'numeric_features_EB015N', 'numeric_features_EC010N', 'numeric_features_EE000N', 'numeric_features_EE007N', 'numeric_features_EH000N', 'numeric_features_EH017N', 'numeric_features_EH018N', 'numeric_features_EI000N', 'numeric_features_EI003N', 'numeric_features_EI005N', 'numeric_features_EI014N', 'numeric_features_EM000N', 'numeric_features_EM019N', 'numeric_features_EN000N', 'numeric_features_EN008N', 'numeric_features_EN016N', 'numeric_features_EO012N', 'numeric_features_EP000N', 'numeric_features_ER000N', 'numeric_features_ER002N', 'numeric_features_EV000N', 'numeric_features_EV001N', 'numeric_features_EV004N', 'numeric_features_EV006N', 'numeric_features_EV011N', 'numeric_features_EV013N', 'numeric_features_EX000N', 'numeric_features_IC000N', 'numeric_features_IC003N', 'numeric_features_IC006N', 'numeric_features_IC007N', 'numeric_features_IC021N', 'numeric_features_IM000N', 'numeric_features_IM009N', 'numeric_features_IM010N', 'numeric_features_IM020N', 'numeric_features_IM022N', 'numeric_features_IM023N', 'numeric_features_IN000N', 'numeric_features_IN002N', 'numeric_features_IN008N', 'numeric_features_IS000N', 'numeric_features_IS004N', 'numeric_features_IS005N', 'numeric_features_IS012N', 'numeric_features_IS013N', 'numeric_features_IS019N', 'numeric_features_IS024N', 'numeric_features_IS025N', 'numeric_features_IU000N', 'numeric_features_IU001N', 'numeric_features_IU011N', 'numeric_features_IU014N', 'numeric_features_IU015N', 'numeric_features_IU016N', 'numeric_features_IU018N', 'numeric_features_IX000N', 'numeric_features_IX017N', 'numeric_features_NA99N', 'numeric_features_OA000N', 'numeric_features_OA001N', 'numeric_features_OA002N', 'numeric_features_OA003N', 'numeric_features_OA004N', 'numeric_features_OB000N', 'numeric_features_OB005N', 'numeric_features_OB006N', 'numeric_features_OC000N', 'numeric_features_PB000O', 'numeric_features_PB033M', 'numeric_features_PB033O', 'numeric_features_PB052O', 'numeric_features_PC000M', 'numeric_features_PC000O', 'numeric_features_PC002M', 'numeric_features_PC002O', 'numeric_features_PC003M', 'numeric_features_PC003O', 'numeric_features_PC008M', 'numeric_features_PC008O', 'numeric_features_PC018M', 'numeric_features_PC018O', 'numeric_features_PC025M', 'numeric_features_PC025O', 'numeric_features_PC031M', 'numeric_features_PC031O', 'numeric_features_PE000M', 'numeric_features_PE000O', 'numeric_features_PE001M', 'numeric_features_PE001O', 'numeric_features_PE035O', 'numeric_features_PE046M', 'numeric_features_PG000M', 'numeric_features_PG000O', 'numeric_features_PG004M', 'numeric_features_PG004O', 'numeric_features_PG006M', 'numeric_features_PG006O', 'numeric_features_PG012O', 'numeric_features_PG026M', 'numeric_features_PG043M', 'numeric_features_PG043O', 'numeric_features_PG047M', 'numeric_features_PG047O', 'numeric_features_PH000O', 'numeric_features_PH034M', 'numeric_features_PH034O', 'numeric_features_PM000M', 'numeric_features_PM000O', 'numeric_features_PM007O', 'numeric_features_PM011M', 'numeric_features_PM011O', 'numeric_features_PM014M', 'numeric_features_PM014O', 'numeric_features_PM015O', 'numeric_features_PM020M', 'numeric_features_PM020O', 'numeric_features_PM021M', 'numeric_features_PM021O', 'numeric_features_PM024M', 'numeric_features_PM024O', 'numeric_features_PM036O', 'numeric_features_PM039M', 'numeric_features_PM039O', 'numeric_features_PM041M', 'numeric_features_PM044M', 'numeric_features_PO000M', 'numeric_features_PO000O', 'numeric_features_PO010M', 'numeric_features_PO010O', 'numeric_features_PO022M', 'numeric_features_PO022O', 'numeric_features_PO027M', 'numeric_features_PO027O', 'numeric_features_PO040M', 'numeric_features_PO040O', 'numeric_features_PO045M', 'numeric_features_PO045O', 'numeric_features_PO050O', 'numeric_features_PS000M', 'numeric_features_PS000O', 'numeric_features_PS009O', 'numeric_features_PS013M', 'numeric_features_PS013O', 'numeric_features_PS016M', 'numeric_features_PS016O', 'numeric_features_PS017O', 'numeric_features_PS023O', 'numeric_features_PS028M', 'numeric_features_PS028O', 'numeric_features_PS032O', 'numeric_features_PS038O', 'numeric_features_PS051O', 'numeric_features_PV000M', 'numeric_features_PV000O', 'numeric_features_PV005M', 'numeric_features_PV005O', 'numeric_features_PV019M', 'numeric_features_PV019O', 'numeric_features_PV029M', 'numeric_features_PV029O', 'numeric_features_PV030O', 'numeric_features_PV037M', 'numeric_features_PV037O', 'numeric_features_PV042O', 'numeric_features_PV048M', 'numeric_features_PV048O', 'numeric_features_PV049M', 'numeric_features_RB000N', 'numeric_features_RB017N', 'numeric_features_RD000N', 'numeric_features_RD001N', 'numeric_features_RD028N', 'numeric_features_RD032N', 'numeric_features_RH000N', 'numeric_features_RH002N', 'numeric_features_RH012N', 'numeric_features_RI000N', 'numeric_features_RI004N', 'numeric_features_RI005N', 'numeric_features_RI006N', 'numeric_features_RI008N', 'numeric_features_RI011N', 'numeric_features_RI013N', 'numeric_features_RI014N', 'numeric_features_RI015N', 'numeric_features_RI016N', 'numeric_features_RI018N', 'numeric_features_RI019N', 'numeric_features_RI022N', 'numeric_features_RI023N', 'numeric_features_RI024N', 'numeric_features_RI025N', 'numeric_features_RI026N', 'numeric_features_RI030N', 'numeric_features_RI031N', 'numeric_features_RR000N', 'numeric_features_RR007N', 'numeric_features_RR009N', 'numeric_features_RR010N', 'numeric_features_RT000N', 'numeric_features_RT003N', 'numeric_features_RT020N', 'numeric_features_RT021N', 'numeric_features_RT033N', 'numeric_features_RX000N', 'numeric_features_RX027N', 'numeric_features_RX029N', 'numeric_features_RX034N', 'numeric_features_TA000N', 'numeric_features_TA002N', 'numeric_features_TA009N', 'numeric_features_TC000N', 'numeric_features_TC003N', 'numeric_features_TC010N', 'numeric_features_TF000N', 'numeric_features_TF015N', 'numeric_features_TL000N', 'numeric_features_TL001N', 'numeric_features_TL004N', 'numeric_features_TL005N', 'numeric_features_TL006N', 'numeric_features_TL012N', 'numeric_features_TL013N', 'numeric_features_TM000N', 'numeric_features_TM011N', 'numeric_features_TM014N', 'numeric_features_TN000N', 'numeric_features_TN007N', 'numeric_features_TN008N', 'numeric_features_TP000N', 'numeric_features_TX000N', 'numeric_features_Z2', 'numeric_features_rx_null', 'numeric_features_E01754130101', 'numeric_features_E01754140101', 'numeric_features_E01754150101', 'numeric_features_E01754160101', 'numeric_features_E01754180101', 'numeric_features_E01754180201', 'numeric_features_E01754190101', 'numeric_features_E01754190201', 'numeric_features_E01754200101', 'numeric_features_E01754200201', 'numeric_features_E01754210101', 'numeric_features_E01754210201', 'numeric_features_E01754230101', 'numeric_features_E01754260101', 'numeric_features_E01754260401', 'numeric_features_E01754270101', 'numeric_features_E01754280101', 'numeric_features_E01754290101', 'numeric_features_E01754300101', 'numeric_features_E01754300201', 'numeric_features_E01754310101', 'numeric_features_E01754340101', 'numeric_features_E01754350101', 'numeric_features_E01754350201', 'numeric_features_E01754430101', 'numeric_features_E01754430201', 'numeric_features_E01754530101', 'numeric_features_E01754540101', 'numeric_features_E01754570101', 'numeric_features_E01754610101', 'numeric_features_E01754620101', 'numeric_features_E01754630101', 'numeric_features_E01754630201', 'numeric_features_E01754630202', 'numeric_features_E01754660101', 'numeric_features_E01754660301', 'numeric_features_E01754670101', 'numeric_features_E01754680101', 'numeric_features_E01754700101', 'numeric_features_E01754730101', 'numeric_features_E01754760101', 'numeric_features_E01754770101', 'numeric_features_E01754770201', 'numeric_features_E01754770202', 'numeric_features_E01754770301', 'numeric_features_E01754810101', 'numeric_features_E01754820101', 'numeric_features_E01754820201', 'numeric_features_E01754830101', 'numeric_features_E01754830201', 'numeric_features_E01754840101', 'numeric_features_E01754850101', 'numeric_features_E01754860101', 'numeric_features_E01754870101', 'numeric_features_E01754870201', 'numeric_features_E01754880101', 'numeric_features_E01754880201', 'numeric_features_E01754890101', 'numeric_features_E01754890201', 'numeric_features_E01754890202', 'numeric_features_E01754910101', 'numeric_features_E01754930101', 'numeric_features_E01754940101', 'numeric_features_E01754950101', 'numeric_features_E01754960101', 'numeric_features_E01754960201', 'numeric_features_E01754970101', 'numeric_features_E01754970201', 'numeric_features_E01754980101', 'numeric_features_E01754990101', 'numeric_features_E01755000101', 'numeric_features_E01755010101', 'numeric_features_E01755030101', 'numeric_features_E01755050101', 'numeric_features_E01755070101', 'numeric_features_E01755090101', 'numeric_features_E01755110101', 'numeric_features_E01755140101', 'numeric_features_E01755150101', 'numeric_features_E01755170101', 'numeric_features_E01755170201', 'numeric_features_E01755190101', 'numeric_features_E01755210101', 'numeric_features_E01755220101', 'numeric_features_E01755240101', 'numeric_features_E01755250101', 'numeric_features_E01755350101', 'numeric_features_E01755390101', 'numeric_features_E01755430101', 'numeric_features_E01755520101', 'numeric_features_E01755530101', 'numeric_features_E01755540101', 'numeric_features_E01755550101', 'numeric_features_E01755560101', 'numeric_features_E01755570101', 'numeric_features_E01755570201', 'numeric_features_E01755580101', 'numeric_features_E01755590101', 'numeric_features_E01755600101', 'numeric_features_E01755610101', 'numeric_features_E01755610201', 'numeric_features_E01755610202', 'numeric_features_E01755610203', 'numeric_features_E01755610301', 'numeric_features_E01755620101', 'numeric_features_E01755620201', 'numeric_features_E01755620202', 'numeric_features_E01755620203', 'numeric_features_E01755630101', 'numeric_features_E01755630201', 'numeric_features_E01755640101', 'numeric_features_E01755640201', 'numeric_features_E01755650101', 'numeric_features_E01755650201', 'numeric_features_E01755650202', 'numeric_features_E01755650203', 'numeric_features_E01755650401', 'numeric_features_E01755660101', 'numeric_features_E01755680101', 'numeric_features_E01755700101', 'numeric_features_E01755720101', 'numeric_features_E01755730101', 'numeric_features_E01755740101', 'numeric_features_E01755740201', 'numeric_features_E01755740202', 'numeric_features_E01755740203', 'numeric_features_E01755740301', 'numeric_features_E01755740303', 'numeric_features_E01755750101', 'numeric_features_E01755760101', 'numeric_features_E01755760201', 'numeric_features_E01755760202', 'numeric_features_E01755760203', 'numeric_features_E01755760204', 'numeric_features_E01755760206', 'numeric_features_E01755760207', 'numeric_features_E01755760209', 'numeric_features_E01755760302', 'numeric_features_E01755760401', 'numeric_features_E01755780101', 'numeric_features_E01755790101', 'numeric_features_E01755790201', 'numeric_features_E01755800101', 'numeric_features_E01755810101', 'numeric_features_E01755820101', 'numeric_features_E01755830101', 'numeric_features_E01755840101', 'numeric_features_E01755860101', 'numeric_features_E01755870101', 'numeric_features_E01755870201', 'numeric_features_E01755870202', 'numeric_features_E01755880101', 'numeric_features_E01755890101', 'numeric_features_E01755900101', 'numeric_features_E01755910101', 'numeric_features_E01755920101', 'numeric_features_E01755940101', 'numeric_features_E01755940401', 'numeric_features_E01755940501', 'numeric_features_E01755950101', 'numeric_features_E01755960101', 'numeric_features_E01755960401', 'numeric_features_E01755970101', 'numeric_features_E01755980101', 'numeric_features_E01755990101', 'numeric_features_E01756010101', 'numeric_features_E01756020101', 'numeric_features_E01756030101', 'numeric_features_E01756040101', 'numeric_features_E01756050101', 'numeric_features_E01756060101', 'numeric_features_E01756070101', 'numeric_features_E01756080101', 'numeric_features_E01756090101', 'numeric_features_E01756100101', 'numeric_features_E01756110101', 'numeric_features_E01756120101', 'numeric_features_E01756130101', 'numeric_features_E01756160101', 'numeric_features_E01756230101', 'numeric_features_E01756250101', 'numeric_features_E01756300101', 'numeric_features_E01756340101', 'numeric_features_E01756370101', 'numeric_features_E01756380101', 'numeric_features_E01756550101', 'numeric_features_E01756550201', 'numeric_features_E01756560101', 'numeric_features_E01756570101', 'numeric_features_E01756570201', 'numeric_features_E01756610101', 'numeric_features_E01756640101', 'numeric_features_E01756650101', 'numeric_features_E01756660101', 'numeric_features_E01756670101', 'numeric_features_E01756690101', 'numeric_features_E01756790101', 'numeric_features_E01756800101', 'numeric_features_E01756810101', 'numeric_features_E01756820101', 'numeric_features_E01756820201', 'numeric_features_E01756820301', 'numeric_features_E01756820302', 'numeric_features_E01756820501', 'numeric_features_E01756830101', 'numeric_features_E01756890101', 'numeric_features_E01756900101', 'numeric_features_E01756900201', 'numeric_features_E01756910101', 'numeric_features_E01756910201', 'numeric_features_E01756910202', 'numeric_features_E01756920101', 'numeric_features_E01756930101', 'numeric_features_E01756930501', 'numeric_features_E01756940101', 'numeric_features_E01756940201', 'numeric_features_E01756950101', 'numeric_features_E01756960101', 'numeric_features_E01756980101', 'numeric_features_E01757000101', 'numeric_features_E01757000201', 'numeric_features_E01757050101', 'numeric_features_E01757060101', 'numeric_features_E01757100101', 'numeric_features_E01757120101', 'numeric_features_E01757130101', 'numeric_features_E01757160101', 'numeric_features_E01757190101', 'numeric_features_E01757200101', 'numeric_features_E01757220101', 'numeric_features_E01757220201', 'numeric_features_E01757220202', 'numeric_features_E01757220203', 'numeric_features_E01757220204', 'numeric_features_E01757220205', 'numeric_features_E01757220206', 'numeric_features_E01757220207', 'numeric_features_E01757230101', 'numeric_features_E01757230201', 'numeric_features_E01757370101', 'numeric_features_E01757370301', 'numeric_features_E01757380101', 'numeric_features_E01757390101', 'numeric_features_E01757390301', 'numeric_features_E01757400101', 'numeric_features_E01757430101', 'numeric_features_E01757440101', 'numeric_features_E01757450101', 'numeric_features_E01757460101', 'numeric_features_E01757460201', 'numeric_features_E01757460202', 'numeric_features_E01757460301', 'numeric_features_E01757490101', 'numeric_features_E01757500101', 'numeric_features_E01757500201', 'numeric_features_E01757510101', 'numeric_features_E01757520101', 'numeric_features_E01757530101', 'numeric_features_E01757530201', 'numeric_features_E01757540101', 'numeric_features_E01757570101', 'numeric_features_E01757580101', 'numeric_features_E01757590101', 'numeric_features_E01757650101', 'numeric_features_E01757660101', 'numeric_features_E01757680101', 'numeric_features_E01757690101', 'numeric_features_E01757710101', 'numeric_features_E01757750101', 'numeric_features_E01757760101', 'numeric_features_E01757770101', 'numeric_features_E01757790101', 'numeric_features_E01757800101', 'numeric_features_E01757810101', 'numeric_features_E01757820101', 'numeric_features_E01757830101', 'numeric_features_E01757840101', 'numeric_features_E01757840201', 'numeric_features_E01757850101', 'numeric_features_E01757860101', 'numeric_features_E01757900101', 'numeric_features_E01757940101', 'numeric_features_E01757960101', 'numeric_features_E01758000101', 'numeric_features_E01758010101', 'numeric_features_E01758020101', 'numeric_features_E01758050101', 'numeric_features_E01758070101', 'numeric_features_E01758090101', 'numeric_features_E01758100101', 'numeric_features_E01758110101', 'numeric_features_E01758110201', 'numeric_features_E01758120101', 'numeric_features_E01758140101', 'numeric_features_E01758150101', 'numeric_features_E01758180101', 'numeric_features_E01758180201', 'numeric_features_E01758190101', 'numeric_features_E01758200101', 'numeric_features_E01758210101', 'numeric_features_E01758220101', 'numeric_features_E01758230101', 'numeric_features_E01758240101', 'numeric_features_E01758250101', 'numeric_features_E01758250201', 'numeric_features_E01758250203', 'numeric_features_E01758250302', 'numeric_features_E01758260101', 'numeric_features_E01758270101', 'numeric_features_E01758280101', 'numeric_features_E01758310101', 'numeric_features_E01758350101', 'numeric_features_E01758360101', 'numeric_features_E01758360201', 'numeric_features_E01758370101', 'numeric_features_E01758380101', 'numeric_features_E01758390101', 'numeric_features_E01758400101', 'numeric_features_E01758410101', 'numeric_features_E01758420101', 'numeric_features_E01758430101', 'numeric_features_E01758450101', 'numeric_features_E01758470101', 'numeric_features_E01758480101', 'numeric_features_E01758490101', 'numeric_features_E01758540101', 'numeric_features_E01758740101', 'numeric_features_E01758750101', 'numeric_features_E01758770101', 'numeric_features_E01758770201', 'numeric_features_E01758790101', 'numeric_features_E01758810101', 'numeric_features_E01758820101', 'numeric_features_E01758840101', 'numeric_features_E01758850101', 'numeric_features_E01758870101', 'numeric_features_E01758870201', 'numeric_features_E01758890101', 'numeric_features_E01758950101', 'numeric_features_E01758980101', 'numeric_features_E01759000101', 'numeric_features_E01759000201', 'numeric_features_E01759020101', 'numeric_features_E01759030101', 'numeric_features_E01759040101', 'numeric_features_E01759070101', 'numeric_features_E01759080101', 'numeric_features_E01759090101', 'numeric_features_E01759090701', 'numeric_features_E01759110101', 'numeric_features_E01759110201', 'numeric_features_E01759130101', 'numeric_features_E01759130201', 'numeric_features_E01759130401', 'numeric_features_E01759180101', 'numeric_features_E01759280101', 'numeric_features_E01759300101', 'numeric_features_E01759340101', 'numeric_features_E01759350101', 'numeric_features_E01759350201', 'numeric_features_E01759360101', 'numeric_features_E01759370101', 'numeric_features_E01759380101', 'numeric_features_E01759400101', 'numeric_features_E01759410101', 'numeric_features_E01759420101', 'numeric_features_E01759440101', 'numeric_features_E01759450101', 'numeric_features_E01759460101', 'numeric_features_E01759490101', 'numeric_features_E01759500301', 'numeric_features_E01759500701', 'numeric_features_E01759510101', 'numeric_features_E01759510201', 'numeric_features_E01759510301', 'numeric_features_E01759510401', 'numeric_features_E01759520101', 'numeric_features_E01759560101', 'numeric_features_E01759580101', 'numeric_features_E01759630101', 'numeric_features_E01759660101', 'numeric_features_E01759730101', 'numeric_features_E01759800101', 'numeric_features_E01759800201', 'numeric_features_E01779100101', 'numeric_features_E01779130101', 'numeric_features_E01779140101', 'numeric_features_E01783260101', 'numeric_features_E01783690101', 'numeric_features_E01783720101', 'numeric_features_E01783740101', 'numeric_features_E01783750101', 'numeric_features_E01783780101', 'numeric_features_E01784800101', 'numeric_features_E01784800201', 'numeric_features_E01801820101', 'numeric_features_E01801830101', 'numeric_features_E01801850101', 'numeric_features_E01801860101', 'numeric_features_E01801870101', 'numeric_features_E01801900101', 'numeric_features_E01802920101', 'numeric_features_E01808500101', 'numeric_features_E01808510101', 'numeric_features_E01808520101', 'numeric_features_E01808530101', 'numeric_features_E01808540101', 'numeric_features_E01808550101', 'numeric_features_E01808550201', 'numeric_features_E01818110101', 'numeric_features_E01818160101', 'numeric_features_E01821420101', 'numeric_features_E01821490301', 'numeric_features_E01821590101', 'numeric_features_E01826330101', 'numeric_features_E01826350101', 'numeric_features_E01826370101', 'numeric_features_E01826390201', 'numeric_features_E01826390301', 'numeric_features_E01828300101', 'numeric_features_E01829610101', 'numeric_features_E01829650101', 'numeric_features_E01829670101', 'numeric_features_E01833600101', 'numeric_features_E01838880101', 'numeric_features_E01838890101', 'numeric_features_E01838890201', 'numeric_features_E01838900101', 'numeric_features_E01838910101', 'numeric_features_E01838940101', 'numeric_features_E01838950101', 'numeric_features_E01838960301', 'numeric_features_E01838960401', 'numeric_features_E01838960501', 'numeric_features_E01838960502', 'numeric_features_E01838970101', 'numeric_features_E01838980101', 'numeric_features_E01839000201', 'numeric_features_E01839010101', 'numeric_features_E01839050101', 'numeric_features_E01839050501', 'numeric_features_E01839060101', 'numeric_features_E01839070301', 'numeric_features_E01839100101', 'numeric_features_E01839120101', 'numeric_features_E01839160101', 'numeric_features_E01840140101', 'numeric_features_E01840150101', 'numeric_features_E01841440101', 'numeric_features_E01841460101', 'numeric_features_E01841490101', 'numeric_features_E01841660101', 'numeric_features_E01841670101', 'numeric_features_E01841690101', 'numeric_features_E01841720101', 'numeric_features_E01841740101', 'numeric_features_E01843160101', 'numeric_features_E01850080101', 'numeric_features_E01850100101', 'numeric_features_E01855000101', 'numeric_features_E01855020101', 'numeric_features_E01855080101', 'numeric_features_E01861050101', 'numeric_features_E01861060101', 'numeric_features_E01867790101', 'numeric_features_E01870510101', 'numeric_features_E01870550101', 'numeric_features_E01870590101', 'numeric_features_E01901190101', 'numeric_features_E01904800101', 'numeric_features_E01904830101', 'numeric_features_E01904850101', 'numeric_features_E01908520101', 'numeric_features_E01908540101', 'numeric_features_E01908560101', 'numeric_features_E01908580101', 'numeric_features_E01909920101', 'numeric_features_E01909960101', 'numeric_features_E01910000101', 'numeric_features_E01910010101', 'numeric_features_E01910010201', 'numeric_features_E01910010301', 'numeric_features_E01910010302', 'numeric_features_E01910010501', 'numeric_features_E01910090101', 'numeric_features_E01912560101', 'numeric_features_E01912560201', 'numeric_features_E01912600101', 'numeric_features_E01912610101', 'numeric_features_E01912630101', 'numeric_features_E01912790101', 'numeric_features_E01912810101', 'numeric_features_E01914200101', 'numeric_features_E01914210101', 'numeric_features_E01914930101', 'numeric_features_E01914950101', 'numeric_features_E01915440101', 'numeric_features_E01916230101', 'numeric_features_E01916250101', 'numeric_features_E01916260101', 'numeric_features_E01917310101', 'numeric_features_E01918650101', 'numeric_features_E01918670101', 'numeric_features_E01918720101', 'numeric_features_E01923360101', 'numeric_features_E01923380101', 'numeric_features_E01923390101', 'numeric_features_E01923420101', 'numeric_features_E01925150101', 'numeric_features_E01925160101', 'numeric_features_E01925610101', 'numeric_features_E01925620101', 'numeric_features_E01927010101', 'numeric_features_E01927500101', 'numeric_features_E01927950101', 'numeric_features_E01927970101', 'numeric_features_E01927990101', 'numeric_features_E01928000101', 'numeric_features_E01928000301', 'numeric_features_E01931810101', 'numeric_features_E01932200201', 'numeric_features_E01932200301', 'numeric_features_E01932230101', 'numeric_features_E01932760101', 'numeric_features_E01933380101', 'numeric_features_E01933420101', 'numeric_features_E01934530101', 'numeric_features_E01935430101', 'numeric_features_E01936180101', 'numeric_features_E01937880101', 'numeric_features_E01938000101', 'numeric_features_E01938030101', 'numeric_features_E01939170101', 'numeric_features_E01939560101', 'numeric_features_W00000010101', 'numeric_features_W00000020101', 'numeric_features_W00000060101', 'numeric_features_W00000060201', 'numeric_features_W00000060202', 'numeric_features_W00000060301', 'numeric_features_W00000060302', 'numeric_features_W00000070101', 'numeric_features_W00000080101', 'numeric_features_W00000090101', 'numeric_features_W00000100101', 'numeric_features_W00000120101', 'numeric_features_W00000130101', 'numeric_features_W00000140101', 'numeric_features_W00000150101', 'numeric_features_W00000160101', 'numeric_features_W00000170101', 'numeric_features_W00000210101', 'numeric_features_W00000220101', 'numeric_features_W00000230101', 'numeric_features_W00000260101', 'numeric_features_W00000290101', 'numeric_features_W00000310101', 'numeric_features_W00000320101', 'numeric_features_W00000430101', 'numeric_features_W00000440101', 'numeric_features_W00000470101', 'numeric_features_W00000480101', 'numeric_features_W00000500101', 'numeric_features_W00000520101', 'numeric_features_W00000530101', 'numeric_features_W00000570101', 'numeric_features_W00000620101', 'numeric_features_W00000630101', 'numeric_features_W00000660101', 'numeric_features_W00000680101', 'numeric_features_W00000690101', 'numeric_features_W00000700101', 'numeric_features_W00000710101', 'numeric_features_W00000720101', 'numeric_features_W00000730101', 'numeric_features_W00000750101', 'numeric_features_W00000780101', 'numeric_features_W00000790101', 'numeric_features_W00000810101', 'numeric_features_W00000860101', 'numeric_features_W00000870101', 'numeric_features_W00000920101', 'numeric_features_W00000950101', 'numeric_features_W00000980101', 'numeric_features_W00001010101', 'numeric_features_W00001030101', 'numeric_features_W00001040101', 'numeric_features_W00001100101', 'numeric_features_W00001110101', 'numeric_features_W00001120101', 'numeric_features_W00001130101', 'numeric_features_W00001140101', 'numeric_features_W00001150101', 'numeric_features_W00001160101', 'numeric_features_W00001160201', 'numeric_features_W00001170101', 'numeric_features_W00001240101', 'numeric_features_W00001250101', 'numeric_features_W00001260101', 'numeric_features_01', 'numeric_features_02', 'numeric_features_03', 'numeric_features_04', 'numeric_features_05', 'numeric_features_06', 'numeric_features_07', 'numeric_features_08', 'numeric_features_09', 'numeric_features_10', 'numeric_features_11', 'numeric_features_12', 'numeric_features_13', 'numeric_features_14', 'numeric_features_15', 'numeric_features_16', 'numeric_features_17', 'numeric_features_18', 'numeric_features_19', 'numeric_features_20', 'numeric_features_21', 'numeric_features_22', 'numeric_features_23', 'numeric_features_24', 'numeric_features_25', 'numeric_features_26', 'numeric_features_27', 'numeric_features_28', 'numeric_features_29', 'numeric_features_30', 'numeric_features_32', 'numeric_features_33', 'numeric_features_34', 'numeric_features_35', 'numeric_features_36', 'numeric_features_37', 'numeric_features_38', 'numeric_features_39', 'numeric_features_40', 'numeric_features_41', 'numeric_features_42', 'numeric_features_43', 'numeric_features_44', 'numeric_features_45', 'numeric_features_46', 'numeric_features_47', 'numeric_features_48', 'numeric_features_49', 'numeric_features_50', 'numeric_features_51', 'numeric_features_52', 'numeric_features_53', 'numeric_features_54', 'numeric_features_55', 'numeric_features_56', 'numeric_features_57', 'numeric_features_58', 'numeric_features_59', 'numeric_features_60', 'numeric_features_61', 'numeric_features_62', 'numeric_features_63', 'numeric_features_64', 'numeric_features_65', 'numeric_features_66', 'numeric_features_67', 'numeric_features_68', 'numeric_features_69', 'numeric_features_70', 'numeric_features_71', 'numeric_features_72', 'numeric_features_73', 'numeric_features_74', 'numeric_features_76', 'numeric_features_77', 'numeric_features_78', 'numeric_features_79', 'numeric_features_80', 'numeric_features_81', 'numeric_features_82', 'numeric_features_83', 'numeric_features_84', 'numeric_features_85', 'numeric_features_86', 'numeric_features_87', 'numeric_features_89', 'numeric_features_90', 'numeric_features_91', 'numeric_features_92', 'numeric_features_93', 'numeric_features_94', 'numeric_features_95', 'numeric_features_96', 'numeric_features_97', 'numeric_features_98', 'numeric_features_A0', 'numeric_features_A1', 'numeric_features_A2', 'numeric_features_A3', 'numeric_features_A4', 'numeric_features_A5', 'numeric_features_A6', 'numeric_features_A9', 'numeric_features_B1', 'numeric_features_B2', 'numeric_features_B3', 'numeric_features_B4', 'numeric_features_B5', 'categorical_features_ageCat_Vec_under10', 'categorical_features_ageCat_Vec_18To29', 'categorical_features_ageCat_Vec_10To17', 'categorical_features_ageCat_Vec_30To39', 'categorical_features_ageCat_Vec_50To64', 'categorical_features_ageCat_Vec_40To49', 'categorical_features_ageCat_Vec_missing', 'categorical_features_sex_Vec_female', 'categorical_features_sex_Vec_male', 'categorical_features_sex_Vec_missing', 'categorical_features_race_Vec_white', 'categorical_features_race_Vec_missing', 'categorical_features_race_Vec_black', 'categorical_features_race_Vec_hispanic', 'categorical_features_race_Vec_asian', 'categorical_features_race_Vec_native', 'categorical_features_race_Vec_hawaiian', 'categorical_features_race_Vec_multiracial', 'categorical_features_houseSize_Vec_missing', 'categorical_features_houseSize_Vec_twoToFive', 'categorical_features_houseSize_Vec_single', 'categorical_features_houseSize_Vec_sixorMore', 'categorical_features_fedPovLine_Vec_missing', 'categorical_features_fedPovLine_Vec_0To100', 'categorical_features_fedPovLine_Vec_100To200', 'categorical_features_fedPovLine_Vec_200AndMore', 'categorical_features_speakEnglish_Vec_missing', 'categorical_features_speakEnglish_Vec_yes', 'categorical_features_speakEnglish_Vec_no', 'categorical_features_married_Vec_no', 'categorical_features_married_Vec_missing', 'categorical_features_married_Vec_yes', 'categorical_features_UsCitizen_Vec_yes', 'categorical_features_UsCitizen_Vec_missing', 'categorical_features_UsCitizen_Vec_no', 'categorical_features_ssi_Vec_no', 'categorical_features_ssi_Vec_missing', 'categorical_features_ssi_Vec_yes', 'categorical_features_ssdi_Vec_no', 'categorical_features_ssdi_Vec_missing', 'categorical_features_ssdi_Vec_yes', 'categorical_features_tanf_Vec_no', 'categorical_features_tanf_Vec_missing', 'categorical_features_tanf_Vec_yes', 'categorical_features_disabled_Vec_no', 'categorical_features_disabled_Vec_yes', 'categorical_features_enrollMonth_Vec_Jan', 'categorical_features_enrollMonth_Vec_Mar', 'categorical_features_enrollMonth_Vec_Feb', 'categorical_features_enrollMonth_Vec_Aug', 'categorical_features_enrollMonth_Vec_Nov', 'categorical_features_enrollMonth_Vec_Apr', 'categorical_features_enrollMonth_Vec_Dec', 'categorical_features_enrollMonth_Vec_May', 'categorical_features_enrollMonth_Vec_Jul', 'categorical_features_enrollMonth_Vec_Jun', 'categorical_features_enrollMonth_Vec_Oct', 'categorical_features_enrollMonth_Vec_Sep', 'categorical_features_enrollYear_Vec_2017', 'categorical_features_enrollYear_Vec_2018', 'categorical_features_state_Vec_IL', 'categorical_features_state_Vec_PA', 'categorical_features_state_Vec_MI', 'categorical_features_state_Vec_AZ', 'categorical_features_state_Vec_WA', 'categorical_features_state_Vec_IN', 'categorical_features_state_Vec_LA', 'categorical_features_state_Vec_TN', 'categorical_features_state_Vec_MD', 'categorical_features_state_Vec_KY', 'categorical_features_state_Vec_VA', 'categorical_features_state_Vec_AL', 'categorical_features_state_Vec_NM', 'categorical_features_state_Vec_NV', 'categorical_features_state_Vec_MS', 'categorical_features_state_Vec_WV', 'categorical_features_state_Vec_KS', 'categorical_features_state_Vec_UT', 'categorical_features_state_Vec_HI', 'categorical_features_state_Vec_ID', 'categorical_features_state_Vec_MT', 'categorical_features_state_Vec_DE', 'categorical_features_state_Vec_DC', 'categorical_features_state_Vec_ME', 'categorical_features_state_Vec_VT', 'categorical_features_state_Vec_ND', 'categorical_features_state_Vec_WY', 'categorical_features_enrollMonths_Vec_12', 'categorical_features_enrollMonths_Vec_11', 'categorical_features_enrollMonths_Vec_10', 'categorical_features_enrollMonths_Vec_9', 'categorical_features_enrollMonths_Vec_7', 'categorical_features_enrollMonths_Vec_1', 'categorical_features_enrollMonths_Vec_8', 'categorical_features_enrollMonths_Vec_6', 'categorical_features_enrollMonths_Vec_5', 'categorical_features_enrollMonths_Vec_4', 'categorical_features_enrollMonths_Vec_2', 'categorical_features_enrollMonths_Vec_3', 'categorical_features_enrollMonths_Vec_0', 'categorical_features_cov2016yes_Vec_1', 'categorical_features_cov2016yes_Vec_0']

importances = get_pyspark_logistic_regression_feature_importances(final_model, features)

# Display feature importances
for feature, importance in importances.items():
    print(f"{feature}: {importance}")