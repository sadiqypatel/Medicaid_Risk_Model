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

states_to_keep = ["AL", "ME", "MT", "VT", "WY", "IL"]

#outpat
outpat = spark.table("dua_058828_spa240.paper1_stage2_outpatient_12month")
print((outpat.count(), len(outpat.columns)))
outpat = outpat.filter(outpat.pre==1)
outpat = outpat.filter(col("state").isin(states_to_keep))
print((outpat.count(), len(outpat.columns)))
outpat = outpat.select("beneID","state","CLM_ID","SRVC_BGN_DT","DGNS_CD_1")
print(outpat.printSchema())

inpatient = spark.table("dua_058828_spa240.paper1_stage2_inpatient_12month")
print((inpatient.count(), len(inpatient.columns)))
inpatient = inpatient.filter(inpatient.pre==1)
inpatient = inpatient.filter(col("state").isin(states_to_keep))
print((inpatient.count(), len(inpatient.columns)))
inpatient = inpatient.select("beneID","state","CLM_ID","SRVC_BGN_DT","DGNS_CD_1")
print(inpatient.printSchema())

pharmacy = spark.table("dua_058828_spa240.paper1_stage2_pharm_12month")
print((pharmacy.count(), len(pharmacy.columns)))
pharmacy = pharmacy.filter(pharmacy.pre==1)
pharmacy = pharmacy.filter(col("state").isin(states_to_keep))
print((pharmacy.count(), len(pharmacy.columns)))
pharmacy = pharmacy.select("beneID","state","CLM_ID","RX_FILL_DT","NDC")
print(pharmacy.printSchema())

# COMMAND ----------

diagnosis = outpat.union(inpatient)
diagnosis.registerTempTable("connections")

cdps_df = spark.sql('''
SELECT DISTINCT beneID, state, SRVC_BGN_DT, DGNS_CD_1
FROM connections;
''')

cdps_df = cdps_df.withColumnRenamed("beneID", "Recipient_ID").withColumnRenamed("DGNS_CD_1", "DIAG_ID")
cdps_df = cdps_df.drop("SRVC_BGN_DT")
cdps_df.show(5)

# COMMAND ----------

pharmacy.registerTempTable("connections")

mxrx_df = spark.sql('''
SELECT DISTINCT beneID, state, RX_FILL_DT, NDC
FROM connections;
''')

mxrx_df = mxrx_df.withColumnRenamed("beneID", "Recipient_ID").withColumnRenamed("NDC", "MRXR_ID")
mxrx_df = mxrx_df.drop("RX_FILL_DT")
mxrx_df.show(5)

# COMMAND ----------

#predictor 1: patient characteristics
from pyspark.sql.functions import col, floor

member_df = spark.table("dua_058828_spa240.demographic_stage2")
states_to_keep = ["AL", "ME", "MT", "VT", "WY", "IL"]
member_df = member_df.filter(col("state").isin(states_to_keep))

member_df = member_df.select('beneID','state','age','sex','disabled')
member_df = member_df.withColumn("age", floor(col("age")).cast("integer"))
member_df = member_df.withColumn("male", when(col("sex") == "male", 1)
                            .when(col("sex") == "female", 0)
                            .otherwise(None))
member_df = member_df.withColumn("ssi", when(col("disabled") == "yes", 1)
                            .when(col("disabled") == "no", 0)
                            .otherwise(None))
member_df = member_df.drop('sex','disabled')
member_df = member_df.withColumnRenamed("beneID", "Recipient_ID")

print((member_df.count(), len(member_df.columns)))
member_df.show(5)

# COMMAND ----------

# Import the CDPS format file and create the 'cdpsfmt_7.0' table
mrxrfmt_file_path = "dbfs:/mnt/dua/dua_058828/SPA240/files/mrxrfmt_7.0_V2.txt"
mrxrfmt_table_name = "mxrxfmt_70"

mrxr_crosswalk = spark.read.format("csv").option("header", "true").option("delimiter", "\t").load(mrxrfmt_file_path)
mrxr_crosswalk.createOrReplaceTempView(mrxrfmt_table_name)
mrxr_crosswalk = mrxr_crosswalk.drop("Class", "Rank")
mrxr_crosswalk.show(5)

# Count the distinct categories in the column
distinct_count = mrxr_crosswalk.select(countDistinct("MRXR_Group").alias("total_count")).first()["total_count"]

# Print the result
print(distinct_count)

# COMMAND ----------

# Import the CDPS format file and create the 'cdpsfmt_7.0' table
from pyspark.sql.functions import countDistinct

cdpsfmt_file_path = "dbfs:/mnt/dua/dua_058828/SPA240/files/cdpsfmt_7.0_V2.txt"
cdpsfmt_table_name = "cdpsfmt_70"

cdps_crosswalk = spark.read.format("csv").option("header", "true").option("delimiter", "\t").load(cdpsfmt_file_path)
cdps_crosswalk.createOrReplaceTempView(cdpsfmt_table_name)
cdps_crosswalk = cdps_crosswalk.withColumnRenamed("Diag_ID", "DIAG_ID")
cdps_crosswalk = cdps_crosswalk.drop("Class", "Rank")
cdps_crosswalk.show(2)

# Count the distinct categories in the column
distinct_count = cdps_crosswalk.select(countDistinct("Diag_Group").alias("total_count")).first()["total_count"]

# Print the result
print(distinct_count)

# COMMAND ----------

cdps_df.show(1)
mxrx_df.show(1)
mrxr_crosswalk.show(1)
cdps_crosswalk.show(1)
member_df.show(1)

# COMMAND ----------

from pyspark.sql.functions import col

# Insert data into 'CDPS_Step_1' DataFrame by joining 'tmp_indx' and '[cdpsfmt_7.0]' DataFrames
cdps_df1 = cdps_df.join(cdps_crosswalk, on="DIAG_ID", how="inner").orderBy("Recipient_ID", "state", "DIAG_ID") 
cdps_df1.show(1)
cdps_df1 = cdps_df1.drop("DIAG_ID", "Class", "Rank")
print((cdps_df.count(), len(cdps_df.columns)))
print((cdps_df1.count(), len(cdps_df1.columns)))

# Count the distinct categories in the column
distinct_count = cdps_df1.select(countDistinct("Diag_Group").alias("total_count")).first()["total_count"]

# Print the result
print(distinct_count)
cdps_df1.show(1)                 

# COMMAND ----------

# Pivot the dataframe and create dummy indicator columns
pivot_df = cdps_df1.groupBy("Recipient_ID", "state").pivot("Diag_Group").agg({"Diag_Group": "count"}).fillna(0)

# Rename the columns to the corresponding Diag_Group values
for diag_group in pivot_df.columns[2:]:
    pivot_df = pivot_df.withColumnRenamed(diag_group, diag_group.strip())
print((pivot_df.count(), len(pivot_df.columns)))

# Convert the values to 0/1 indicator
for diag_group in pivot_df.columns[2:]:
    pivot_df = pivot_df.withColumn(diag_group, (col(diag_group) > 0).cast("int"))
print((pivot_df.count(), len(pivot_df.columns)))

pivot_df.show()

# COMMAND ----------

from pyspark.sql.functions import max, col

# Aggregate the dataframe by Recipient_ID and state, taking the maximum value for each column
cdps_final = pivot_df.groupBy("Recipient_ID", "state").agg(
    *[max(col(c)).alias(c.replace("max(", "").replace(")", "")) for c in pivot_df.columns[2:]]
)

# Show the resulting dataframe
cdps_final.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Insert data into 'CDPS_Step_1' DataFrame by joining 'tmp_indx' and '[cdpsfmt_7.0]' DataFrames
mxrx_df1 = mxrx_df.join(mrxr_crosswalk, on="MRXR_ID", how="inner").orderBy("Recipient_ID", "state", "MRXR_ID") 
mxrx_df1.show(1)

print((mxrx_df.count(), len(mxrx_df.columns)))
print((mxrx_df1.count(), len(mxrx_df1.columns)))

# Count the distinct categories in the column
distinct_count = mxrx_df1.select(countDistinct("MRXR_Group").alias("total_count")).first()["total_count"]

# Print the result
print(distinct_count)
mxrx_df1.show(1)   

# COMMAND ----------

# Pivot the dataframe and create dummy indicator columns
pivot_df = mxrx_df1.groupBy("Recipient_ID", "state").pivot("MRXR_Group").agg({"MRXR_Group": "count"}).fillna(0)

# Rename the columns to the corresponding Diag_Group values
for mrxr_group in pivot_df.columns[2:]:
    pivot_df = pivot_df.withColumnRenamed(mrxr_group, mrxr_group.strip())
print((pivot_df.count(), len(pivot_df.columns)))

# Convert the values to 0/1 indicator
for mrxr_group in pivot_df.columns[2:]:
    pivot_df = pivot_df.withColumn(mrxr_group, (col(mrxr_group) > 0).cast("int"))
print((pivot_df.count(), len(pivot_df.columns)))

pivot_df.show()

# COMMAND ----------

from pyspark.sql.functions import max, col

# Aggregate the dataframe by Recipient_ID and state, taking the maximum value for each column
mxrx_final = pivot_df.groupBy("Recipient_ID", "state").agg(
    *[max(col(c)).alias(c.replace("max(", "").replace(")", "")) for c in pivot_df.columns[2:]]
)

# Show the resulting dataframe
print((mxrx_final.count(), len(mxrx_final.columns)))
mxrx_final.show()

# COMMAND ----------

# Step 3: Insert data into CDPS_Result
member_df = member_df.select("Recipient_ID","state", "age", "male", "SSI") \
    .distinct()
member_df.show(5)

# Update demographic information in CDPS_Result
member_final = member_df.withColumn("a_under1", when(col("age") < 1, 1).otherwise(0))\
     .withColumn("a_1_4", when((col("age") >= 1) & (col("age") <= 4), 1).otherwise(0))\
     .withColumn("a_5_14m", when((col("age") >= 5) & (col("age") <= 14) & (col("male") == 1), 1).otherwise(0))\
     .withColumn("a_5_14f", when((col("age") >= 5) & (col("age") <= 14) & (col("male") == 0), 1).otherwise(0))\
     .withColumn("a_15_24m", when((col("age") >= 15) & (col("age") <= 24) & (col("male") == 1), 1).otherwise(0))\
     .withColumn("a_15_24f", when((col("age") >= 15) & (col("age") <= 24) & (col("male") == 0), 1).otherwise(0))\
     .withColumn("a_25_44m", when((col("age") >= 25) & (col("age") <= 44) & (col("male") == 1), 1).otherwise(0))\
     .withColumn("a_25_44f", when((col("age") >= 25) & (col("age") <= 44) & (col("male") == 0), 1).otherwise(0))\
     .withColumn("a_45_64m", when((col("age") >= 45) & (col("age") <= 64) & (col("male") == 1), 1).otherwise(0))\
     .withColumn("a_45_64f", when((col("age") >= 45) & (col("age") <= 64) & (col("male") == 0), 1).otherwise(0))\
     .withColumn("a_65", when((col("age") >= 65), 1).otherwise(0))

print((member_final.count(), len(member_final.columns)))
member_final.show()   

# COMMAND ----------

# Perform left join on Recipient_ID and state
final_data = member_final.join(mxrx_final, ["Recipient_ID", "state"], "left").join(cdps_final, ["Recipient_ID", "state"], "left")

# Fill null values with 0 in the joined dataframe
final_data = final_data.fillna(0)

print((member_final.count(), len(member_final.columns)))
print((final_data.count(), len(final_data.columns)))

# Show the resulting dataframe
final_data.show()
print(final_data.printSchema())

# COMMAND ----------

from pyspark.sql.functions import when, col

final_data = final_data.withColumn("CARM", when(col("MRX1") == 1, 1).otherwise(col("CARM")))
final_data = final_data.withColumn("CAREL", when(col("MRX2") == 1, 1).otherwise(col("CAREL")))
final_data = final_data.withColumn("CARM", when(col("CARVH") == 1, 0).otherwise(col("CARM")))
final_data = final_data.withColumn("CARL", when(col("CARVH") == 1, 0).otherwise(col("CARL")))
final_data = final_data.withColumn("CAREL", when(col("CARVH") == 1, 0).otherwise(col("CAREL")))
final_data = final_data.withColumn("CARL", when(col("CARM") == 1, 0).otherwise(col("CARL")))
final_data = final_data.withColumn("CAREL", when(col("CARM") == 1, 0).otherwise(col("CAREL")))
final_data = final_data.withColumn("CAREL", when(col("CARL") == 1, 0).otherwise(col("CAREL")))
final_data = final_data.withColumn("PSYL", when((col("MRX3") == 1) & (col("PSYH") + col("PSYM") == 0), 1).otherwise(col("PSYL")))
final_data = final_data.withColumn("DIA2", when((col("MRX4") == 1) & (col("DIA1") == 0) & (col("age") >= 19), 1).otherwise(col("DIA2")))
final_data = final_data.withColumn("HEMEH", when(col("MRX5") == 1, 1).otherwise(col("HEMEH")))
final_data = final_data.withColumn("HEMVH", when(col("MRX5") == 1, 0).otherwise(col("HEMVH")))
final_data = final_data.withColumn("HEMM", when(col("MRX5") == 1, 0).otherwise(col("HEMM")))
final_data = final_data.withColumn("HEML", when(col("MRX5") == 1, 0).otherwise(col("HEML")))
final_data = final_data.withColumn("INFM", when(col("MRX6") == 1, 1).otherwise(col("INFM")))
final_data = final_data.withColumn("INFM", when(col("MRX7") == 1, 1).otherwise(col("INFM")))
final_data = final_data.withColumn("INFH", when(col("MRX8") == 1, 1).otherwise(col("INFH")))
final_data = final_data.withColumn("INFL", when(col("MRX15") == 1, 1).otherwise(col("INFL")))
final_data = final_data.withColumn("INFH", when(col("INFVH") == 1, 0).otherwise(col("INFH")))
final_data = final_data.withColumn("INFM", when(col("INFVH") == 1, 0).otherwise(col("INFM")))
final_data = final_data.withColumn("INFL", when(col("INFVH") == 1, 0).otherwise(col("INFL")))
final_data = final_data.withColumn("INFM", when(col("INFH") == 1, 0).otherwise(col("INFM")))
final_data = final_data.withColumn("INFL", when(col("INFH") == 1, 0).otherwise(col("INFL")))
final_data = final_data.withColumn("INFL", when(col("INFM") == 1, 0).otherwise(col("INFL")))
final_data = final_data.withColumn("SKCM", when(col("MRX9") == 1, 1).otherwise(col("SKCM")))
final_data = final_data.withColumn("SKCL", when(col("MRX9") == 1, 0).otherwise(col("SKCL")))
final_data = final_data.withColumn("SKCVL", when(col("MRX9") == 1, 0).otherwise(col("SKCVL")))
final_data = final_data.withColumn("CANM", when((col("MRX10") == 1) & (col("CANVH") + col("CANH") == 0), 1).otherwise(col("CANM")))
final_data = final_data.withColumn("CANL", when((col("MRX10") == 1) & (col("CANVH") + col("CANH") == 0), 0).otherwise(col("CANL")))
final_data = final_data.withColumn("CNSH", when(col("MRX11") == 1, 1).otherwise(col("CNSH")))
final_data = final_data.withColumn("CNSL", when((col("MRX12") + col("MRX14")) > 0, 1).otherwise(col("CNSL")))
final_data = final_data.withColumn("CNSM", when(col("CNSH") == 1, 0).otherwise(col("CNSM")))
final_data = final_data.withColumn("CNSL", when(col("CNSM") == 1, 0).otherwise(col("CNSL")))

# COMMAND ----------

# Group by the categorical variable and count the occurrences
final_data.groupBy('CARM').count().show()
final_data.groupBy('CAREL').count().show()
final_data.groupBy('PSYL').count().show()
final_data.groupBy('DIA2').count().show()
final_data.groupBy('SKCM').count().show()
final_data.groupBy('SKCL').count().show()
final_data.groupBy('INFM').count().show()
final_data.groupBy('INFL').count().show()
final_data.groupBy('HEMVH').count().show()


# COMMAND ----------

# from pyspark.sql.functions import lit

# final_data = final_data.withColumn("CCARVH", when((col("SSI") == 1) & (col("age") < 19), col("CARVH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CCARM", when((col("SSI") == 1) & (col("age") < 19), col("CARM")).otherwise(lit(None)))
# final_data = final_data.withColumn("CPSYH", when((col("SSI") == 1) & (col("age") < 19), col("PSYH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CPSYM", when((col("SSI") == 1) & (col("age") < 19), col("PSYM")).otherwise(lit(None)))
# final_data = final_data.withColumn("CCNSH", when((col("SSI") == 1) & (col("age") < 19), col("CNSH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CPULVH", when((col("SSI") == 1) & (col("age") < 19), col("PULVH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CGIH", when((col("SSI") == 1) & (col("age") < 19), col("GIH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CDIA1", when((col("SSI") == 1) & (col("age") < 19), col("DIA1")).otherwise(lit(None)))
# final_data = final_data.withColumn("CRENEH", when((col("SSI") == 1) & (col("age") < 19), col("RENEH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CSUBL", when((col("SSI") == 1) & (col("age") < 19), col("SUBL")).otherwise(lit(None)))
# final_data = final_data.withColumn("CSUBVL", when((col("SSI") == 1) & (col("age") < 19), col("SUBVL")).otherwise(lit(None)))
# final_data = final_data.withColumn("CDDM", when((col("SSI") == 1) & (col("age") < 19), col("DDM")).otherwise(lit(None)))
# final_data = final_data.withColumn("CMETH", when((col("SSI") == 1) & (col("age") < 19), col("METH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CMETM", when((col("SSI") == 1) & (col("age") < 19), col("METM")).otherwise(lit(None)))
# final_data = final_data.withColumn("CINFVH", when((col("SSI") == 1) & (col("age") < 19), col("INFVH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CINFH", when((col("SSI") == 1) & (col("age") < 19), col("INFH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CINFM", when((col("SSI") == 1) & (col("age") < 19), col("INFM")).otherwise(lit(None)))
# final_data = final_data.withColumn("CHEMEH", when((col("SSI") == 1) & (col("age") < 19), col("HEMEH")).otherwise(lit(None)))
# final_data = final_data.withColumn("CHEMVH", when((col("SSI") == 1) & (col("age") < 19), col("HEMVH")).otherwise(lit(None)))

# COMMAND ----------

print(final_data.printSchema())

# COMMAND ----------

cost = spark.table("dua_058828_spa240.paper1_stage2_final_data_2M")
cost = cost.select("beneID","state","total_cost")
cost = cost.withColumnRenamed("beneID", "Recipient_ID")
cost.show()

# COMMAND ----------

final_sample = final_data.join(cost, on=["Recipient_ID","state"], how="left")
print((final_data.count(), len(final_data.columns)))
print((final_sample.count(), len(final_sample.columns)))
print((cost.count(), len(cost.columns)))

# COMMAND ----------

summary = final_sample.describe("total_cost")
summary.show()

# COMMAND ----------

final_sample.write.saveAsTable("dua_058828_spa240.paper1_cdps_analysis", mode='overwrite')

# COMMAND ----------

final_sample = spark.table("dua_058828_spa240.paper1_cdps_analysis")
print((final_sample.count(), len(final_sample.columns)))
adults = final_sample.filter((col("age") > 17) & (col("SSI") == 0))
print((adults.count(), len(adults.columns)))
kids   = final_sample.filter((col("age") < 18) & (col("SSI") == 0))
print((kids.count(), len(kids.columns)))
disabled   = final_sample.filter((col("SSI") == 1))
print((disabled.count(), len(disabled.columns)))

# COMMAND ----------

kids.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_kids_2M_new", mode='overwrite') 
adults.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_adults_2M_new", mode='overwrite') 
disabled.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_disabled_2M_new", mode='overwrite')

# COMMAND ----------

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_cdps_analysis")

# COMMAND ----------

column_list = final_data.columns
print(column_list)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_cdps_analysis")

# List of column names
column_list = ['a_under1', 'a_1_4', 'a_5_14m', 'a_5_14f', 'a_15_24m', 'a_15_24f', 'a_25_44m', 'a_25_44f', 'a_45_64m', 'a_45_64f', 'a_65', 'MRX1', 'MRX10', 'MRX11', 'MRX12', 'MRX13', 'MRX14', 'MRX15', 'MRX2', 'MRX3', 'MRX4', 'MRX5', 'MRX6', 'MRX7', 'MRX8', 'MRX9', 'BABY1', 'BABY2', 'BABY3', 'BABY4', 'BABY5', 'BABY6', 'BABY7', 'BABY8', 'CANH', 'CANL', 'CANM', 'CANVH', 'CAREL', 'CARL', 'CARM', 'CARVH', 'CERM', 'CNSH', 'CNSL', 'CNSM', 'DDL', 'DDM', 'DIA1', 'DIA2', 'EYEL', 'EYEVL', 'GENEL', 'GIH', 'GIL', 'GIM', 'HEMEH', 'HEML', 'HEMM', 'HEMVH', 'INFH', 'INFL', 'INFM', 'INFVH', 'METH', 'METM', 'METVL', 'PRGCMP', 'PRGINC', 'PSYH', 'PSYL', 'PSYM', 'PULH', 'PULL', 'PULM', 'PULVH', 'RENEH', 'RENL', 'RENM', 'SKCL', 'SKCM', 'SKCVL', 'SKNH', 'SKNL', 'SKNVL', 'SUBL', 'SUBVL']

# Create VectorAssembler
assembler = VectorAssembler(inputCols=column_list, outputCol='features')

# Transform the DataFrame
vector_df = assembler.transform(df)

# Select the vector column
vector_column = vector_df.select('features')

# Show the result
#display(vector_df)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

train_data, test_data = vector_df.randomSplit([0.8, 0.2], seed=2342)

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

from pyspark.ml.feature import VectorAssembler

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_cdps_kids_2M_new")

# List of column names
column_list = ['a_under1', 'a_1_4', 'a_5_14m', 'a_5_14f', 'a_15_24m', 'a_15_24f', 'a_25_44m', 'a_25_44f', 'a_45_64m', 'a_45_64f', 'a_65', 'MRX1', 'MRX10', 'MRX11', 'MRX12', 'MRX13', 'MRX14', 'MRX15', 'MRX2', 'MRX3', 'MRX4', 'MRX5', 'MRX6', 'MRX7', 'MRX8', 'MRX9', 'BABY1', 'BABY2', 'BABY3', 'BABY4', 'BABY5', 'BABY6', 'BABY7', 'BABY8', 'CANH', 'CANL', 'CANM', 'CANVH', 'CAREL', 'CARL', 'CARM', 'CARVH', 'CERM', 'CNSH', 'CNSL', 'CNSM', 'DDL', 'DDM', 'DIA1', 'DIA2', 'EYEL', 'EYEVL', 'GENEL', 'GIH', 'GIL', 'GIM', 'HEMEH', 'HEML', 'HEMM', 'HEMVH', 'INFH', 'INFL', 'INFM', 'INFVH', 'METH', 'METM', 'METVL', 'PRGCMP', 'PRGINC', 'PSYH', 'PSYL', 'PSYM', 'PULH', 'PULL', 'PULM', 'PULVH', 'RENEH', 'RENL', 'RENM', 'SKCL', 'SKCM', 'SKCVL', 'SKNH', 'SKNL', 'SKNVL', 'SUBL', 'SUBVL']

# Create VectorAssembler
assembler = VectorAssembler(inputCols=column_list, outputCol='features')

# Transform the DataFrame
vector_df = assembler.transform(df)

# Select the vector column
vector_column = vector_df.select('features')

# Show the result
#display(vector_df)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

train_data, test_data = vector_df.randomSplit([0.8, 0.2], seed=2342)

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

predictions.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_kids_2M_pred", mode='overwrite')

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_cdps_adults_2M_new")

# List of column names
column_list = ['a_under1', 'a_1_4', 'a_5_14m', 'a_5_14f', 'a_15_24m', 'a_15_24f', 'a_25_44m', 'a_25_44f', 'a_45_64m', 'a_45_64f', 'a_65', 'MRX1', 'MRX10', 'MRX11', 'MRX12', 'MRX13', 'MRX14', 'MRX15', 'MRX2', 'MRX3', 'MRX4', 'MRX5', 'MRX6', 'MRX7', 'MRX8', 'MRX9', 'BABY1', 'BABY2', 'BABY3', 'BABY4', 'BABY5', 'BABY6', 'BABY7', 'BABY8', 'CANH', 'CANL', 'CANM', 'CANVH', 'CAREL', 'CARL', 'CARM', 'CARVH', 'CERM', 'CNSH', 'CNSL', 'CNSM', 'DDL', 'DDM', 'DIA1', 'DIA2', 'EYEL', 'EYEVL', 'GENEL', 'GIH', 'GIL', 'GIM', 'HEMEH', 'HEML', 'HEMM', 'HEMVH', 'INFH', 'INFL', 'INFM', 'INFVH', 'METH', 'METM', 'METVL', 'PRGCMP', 'PRGINC', 'PSYH', 'PSYL', 'PSYM', 'PULH', 'PULL', 'PULM', 'PULVH', 'RENEH', 'RENL', 'RENM', 'SKCL', 'SKCM', 'SKCVL', 'SKNH', 'SKNL', 'SKNVL', 'SUBL', 'SUBVL']

# Create VectorAssembler
assembler = VectorAssembler(inputCols=column_list, outputCol='features')

# Transform the DataFrame
vector_df = assembler.transform(df)

# Select the vector column
vector_column = vector_df.select('features')

# Show the result
#display(vector_df)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

train_data, test_data = vector_df.randomSplit([0.8, 0.2], seed=2342)

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

predictions.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_adults_2M_pred", mode='overwrite')

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Assuming you have already loaded your data into a DataFrame called 'data'
# Select the relevant columns
df = spark.table("dua_058828_spa240.paper1_stage2_cdps_disabled_2M_new")

# List of column names
column_list = ['a_under1', 'a_1_4', 'a_5_14m', 'a_5_14f', 'a_15_24m', 'a_15_24f', 'a_25_44m', 'a_25_44f', 'a_45_64m', 'a_45_64f', 'a_65', 'MRX1', 'MRX10', 'MRX11', 'MRX12', 'MRX13', 'MRX14', 'MRX15', 'MRX2', 'MRX3', 'MRX4', 'MRX5', 'MRX6', 'MRX7', 'MRX8', 'MRX9', 'BABY1', 'BABY2', 'BABY3', 'BABY4', 'BABY5', 'BABY6', 'BABY7', 'BABY8', 'CANH', 'CANL', 'CANM', 'CANVH', 'CAREL', 'CARL', 'CARM', 'CARVH', 'CERM', 'CNSH', 'CNSL', 'CNSM', 'DDL', 'DDM', 'DIA1', 'DIA2', 'EYEL', 'EYEVL', 'GENEL', 'GIH', 'GIL', 'GIM', 'HEMEH', 'HEML', 'HEMM', 'HEMVH', 'INFH', 'INFL', 'INFM', 'INFVH', 'METH', 'METM', 'METVL', 'PRGCMP', 'PRGINC', 'PSYH', 'PSYL', 'PSYM', 'PULH', 'PULL', 'PULM', 'PULVH', 'RENEH', 'RENL', 'RENM', 'SKCL', 'SKCM', 'SKCVL', 'SKNH', 'SKNL', 'SKNVL', 'SUBL', 'SUBVL']

# Create VectorAssembler
assembler = VectorAssembler(inputCols=column_list, outputCol='features')

# Transform the DataFrame
vector_df = assembler.transform(df)

# Select the vector column
vector_column = vector_df.select('features')

# Show the result
#display(vector_df)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

train_data, test_data = vector_df.randomSplit([0.8, 0.2], seed=2342)

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

predictions.write.saveAsTable("dua_058828_spa240.paper1_stage2_cdps_disabled_2M_pred", mode='overwrite')

# COMMAND ----------

