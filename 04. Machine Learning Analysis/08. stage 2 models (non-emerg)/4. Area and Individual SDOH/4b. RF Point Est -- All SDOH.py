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
df = df.withColumn("non_emerg_binary", when(col("avoid_acute_post") > 0, 1).otherwise(0))
df.groupBy("non_emerg_binary").count().orderBy(col("count").desc()).show()

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

sampled_df = sampled_df.select("beneID", "state", "features", "non_emerg_binary")
print(sampled_df.count())
sampled_df.groupBy("non_emerg_binary").count().show()

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

# Loop through each hyperparameter combination in the param_grid

# Create an instance of the XGBoostClassifier
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="non_emerg_binary")

# Create an instance of the BinaryClassificationEvaluator
auc_evaluator = BinaryClassificationEvaluator(
    rawPredictionCol="rawPrediction",  # The column containing raw predictions (e.g., rawPrediction)
    labelCol="non_emerg_binary",  # The column containing true labels (e.g., label)
    metricName="areaUnderROC"  # The metric to evaluate (e.g., area under the ROC curve)
)

# Create an instance of the MulticlassClassificationEvaluator
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="non_emerg_binary",  # The column containing true labels (e.g., label)
    predictionCol="prediction",  # The column containing predicted labels (e.g., prediction)
    metricName="accuracy"  # The metric to evaluate (e.g., accuracy)
)

best_hyperparams = [
     {'minInstancesPerNode': 1,  'featureSubsetStrategy': "sqrt", 'maxDepth': 20, 'numTrees': 10}
]

import ast
best_hyper = best_hyperparams[0]
print(best_hyperparams)
print(best_hyper)

rf = RandomForestClassifier(featuresCol="features", labelCol="non_emerg_binary",  **best_hyper)
model = rf.fit(train_df)
predictions = model.transform(test_df)

# Evaluate the performance of the best model
auc = auc_evaluator.evaluate(predictions)
tp = predictions.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 1.0)).count()
tn = predictions.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 0.0)).count()
fp = predictions.filter((col('prediction') == 1.0) & (col('non_emerg_binary') == 0.0)).count()
fn = predictions.filter((col('prediction') == 0.0) & (col('non_emerg_binary') == 1.0)).count()

sensitivity = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
ppv = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
npv = tn / (tn + fn) if (tn + fn) > 0.0 else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0.0 else 0.0
mcc = calculate_mcc(tn, fp, fn, tp)    
precision = tp / (tp + fp)
recall = tp / (tp + fn) 
f1_score = (2 * precision * recall) / (precision + recall)

print(auc)
print(accuracy)
print(mcc)
print(f1_score)
print(sensitivity)
print(specificity)
print(ppv)
print(npv)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT

# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])

# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
test_logReg_withReg = predictions.withColumn('probability_col', extract_second_element('probability'))

# Select specific columns
test_logReg_withReg = test_logReg_withReg.select('beneID', 'state', 'probability_col', 'probability', 'rawPrediction', 'prediction', 'non_emerg_binary')

# Display the selected columns
test_logReg_withReg.show(25)

# COMMAND ----------

test_logReg_withReg.write.saveAsTable("dua_058828_spa240.test_rf_stage2_nonemerg", mode="overwrite")