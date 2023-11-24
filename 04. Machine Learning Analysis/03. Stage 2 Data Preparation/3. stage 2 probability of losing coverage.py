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

# COMMAND ----------

df = spark.table("dua_058828_spa240.demographic_stage2")
# Count unique values in df1
unique_values_df = df.select("state").distinct().count()

# Print the results
print("Unique values in df1: ", unique_values_df)

print(df.count())
print(df.printSchema())

# COMMAND ----------

# Assume 'df' is the original DataFrame with 18M rows

# Calculate the fraction to sample in order to get approximately 500k rows
fraction = df.count() / df.count()

#fraction = 10000000 / df.count()

# Take a random sample from the DataFrame
sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# Show the number of rows in the sampled DataFrame
print("Number of rows in the sampled DataFrame:", sampled_df.count())

print((sampled_df.count(), len(sampled_df.columns)))

# Count unique values in df1
unique_values_df = sampled_df.select("state").distinct().count()

# Print the results
print("Unique values in df1: ", unique_values_df)

# Continue with your analysis using the sampled DataFrame 'sampled_df'

#sampled_df.show(10)

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
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import expr
from pyspark.ml.linalg import DenseVector
import warnings

warnings.filterwarnings('ignore')
#warnings.filterwarnings('ignore', category=DeprecationWarning)

# Sample data with categorical variables, continuous variable, and binary label "enrolled"
columns = ['ageCat','sex','race','state','houseSize','fedPovLine','speakEnglish','married','UsCitizen','ssi','ssdi','tanf','disabled','enrollMonth','enrollYear','cov2016yes']

#columns = ['ageCat','sex','race','state','houseSize','fedPovLine','speakEnglish','married','UsCitizen','ssi','ssdi','tanf','disabled','enrollMonth','enrollYear','cov2016yes']

# Define transformers and assembler
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index") for c in columns[:]]
encoders = [OneHotEncoder(inputCol=c + "_index", outputCol=c + "_onehot") for c in columns[:]]
assembler = VectorAssembler(inputCols=[c + "_onehot" for c in columns[:]], outputCol="features")
#print(features)

# Create pipeline and transform training data
feature_pipeline = Pipeline(stages=indexers + encoders + [assembler])
transformed_df = feature_pipeline.fit(sampled_df).transform(sampled_df)
print(transformed_df.count())
display(transformed_df)

print((transformed_df.count(), len(transformed_df.columns)))

# COMMAND ----------

import mlflow.pyfunc
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
import pandas as pd
import mlflow.spark
from pyspark.ml import PipelineModel
import mlflow.pyfunc
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import warnings

warnings.filterwarnings("ignore")

logged_model = 'dbfs:/databricks/mlflow-tracking/626315/0836298633ac4392a71a474fdbd6c77b/artifacts/final_model'

# Load model as a Spark UDF (for inferencing)
sadiq_pyfunc = mlflow.pyfunc.spark_udf(spark,logged_model)

# Load model as a PySpark ML model (for developing)
sadiq_spark = mlflow.spark.load_model(logged_model)

# COMMAND ----------

scored = sadiq_spark.transform(transformed_df)

# COMMAND ----------

display(scored)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import VectorUDT
 
# Define a UDF to extract the second element of the vector
@udf(returnType=FloatType())
def extract_second_element(vector):
    return float(vector[1])
 
# Apply the UDF to the 'probability' column and create a new column 'probability_2nd_value'
stage2_probability = scored.withColumn('stage1_lose_coverage_prob', extract_second_element('probability'))
 
# Select specific columns
stage2_probability = stage2_probability.select('beneID', 'state', 'stage1_lose_coverage_prob', 'probability', 'rawPrediction', 'prediction','loseCoverage')
 
# Display the selected columns
display(stage2_probability)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import DataFrameStatFunctions

# Assuming `stage2_probability` is your DataFrame
# Select the columns of interest
selected_columns = ['prediction', 'loseCoverage']
selected_df = stage2_probability.select(*selected_columns)

# Convert column names to strings
column_names = [col_name for col_name in selected_columns]

# Perform the crosstab analysis
crosstab_result = DataFrameStatFunctions(selected_df).crosstab(column_names[0], column_names[1])

# Display the crosstab result
crosstab_result.show()

# COMMAND ----------

stage2_probability.write.saveAsTable("dua_058828_spa240.stage2_probability", mode="overwrite")

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_probability")
print(df.count())

# COMMAND ----------

