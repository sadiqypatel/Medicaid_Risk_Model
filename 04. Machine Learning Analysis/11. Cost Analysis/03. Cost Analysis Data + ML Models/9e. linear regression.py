# Databricks notebook source
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession

# COMMAND ----------

df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
# Print all the features
print("Features:")
for feature in df.columns:
    print(feature)

# COMMAND ----------

display(df)

# COMMAND ----------

df.select("total_cost").describe().show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_baseline", "total_cost")

# Define the number of folds for cross-validation
num_folds = 5

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_baseline", "total_cost")

# Create a VectorAssembler to combine the feature columns into a single vector column
assembler = VectorAssembler(inputCols=["features_baseline"], outputCol="features")

# Transform the DataFrame to include the feature vector column
df = assembler.transform(df)

# Split the data into training and testing sets (80/20 split)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Create a LinearRegression model
lr = LinearRegression(labelCol="total_cost", featuresCol="features")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using a suitable metric (e.g., R2)
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="total_cost", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"R2: {r2}")                 

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_no_sdoh", "total_cost")

# Create a VectorAssembler to combine the feature columns into a single vector column
assembler = VectorAssembler(inputCols=["features_no_sdoh"], outputCol="features")

# Transform the DataFrame to include the feature vector column
df = assembler.transform(df)

# Split the data into training and testing sets (80/20 split)
train_data, test_data = df.randomSplit([0.75, 0.25], seed=42)

# Create a LinearRegression model
lr = LinearRegression(labelCol="total_cost", featuresCol="features")

# Fit the model on the training data
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using a suitable metric (e.g., R2)
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="total_cost", predictionCol="prediction", metricName="r2")
r2 = evaluator.evaluate(predictions)
print(f"R2: {r2}")     

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_no_sdoh", "total_cost")

# Split the data into training and test sets (80/20 split)
train_data, test_data = df.randomSplit([0.8, 0.2], seed=12345)

# Train the linear regression model on the full training data
lr_model = LinearRegression(labelCol="total_cost", featuresCol="features_no_sdoh")
lr_model = lr_model.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
r2 = evaluator.evaluate(predictions)

# Print the R-squared metric
print("R-squared:", r2)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_area_sdoh", "total_cost")

# Split the data into training and test sets (80/20 split)
train_data, test_data = df.randomSplit([0.9, 0.1], seed=123)

# Train the linear regression model on the full training data
lr_model = LinearRegression(labelCol="total_cost", featuresCol="features_area_sdoh")
lr_model = lr_model.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
r2 = evaluator.evaluate(predictions)

# Print the R-squared metric
print("R-squared:", r2)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
df = df.select("features_all_sdoh", "total_cost")

# Split the data into training and test sets (80/20 split)
train_data, test_data = df.randomSplit([0.9, 0.1], seed=12345)

# Train the linear regression model on the full training data
lr_model = LinearRegression(labelCol="total_cost", featuresCol="features_all_sdoh")
lr_model = lr_model.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model using RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="total_cost", metricName="r2")
r2 = evaluator.evaluate(predictions)

# Print the R-squared metric
print("R-squared:", r2)

# COMMAND ----------

