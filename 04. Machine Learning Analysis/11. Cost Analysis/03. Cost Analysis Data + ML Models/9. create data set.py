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

# COMMAND ----------

demo = spark.table("dua_058828_spa240.stage2_cost_analysis_sample")

# Show the merged DataFrame
print((demo.count(), len(demo.columns)))

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_cost_vector_baseline_variables")
df = df.select("beneID", "state", "features")
df1 = df.withColumnRenamed('features', 'features_baseline')
print(df.count())

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_cost_vector_no_SDOH_variables")
df = df.select("beneID", "state", "features")
df2 = df.withColumnRenamed('features', 'features_no_sdoh')
print(df.count())

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_cost_vector_area_SDOH_variables")
df = df.select("beneID", "state", "features")
df3 = df.withColumnRenamed('features', 'features_area_sdoh')
print(df.count())

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_cost_vector_all_variables")
df = df.select("beneID", "state", "features")
df4 = df.withColumnRenamed('features', 'features_all_sdoh')
print(df.count())

# COMMAND ----------

# Merge df1, df2, df3, and df4
df = df1.join(df2, ['beneID', 'state'], 'inner') \
    .join(df3, ['beneID', 'state'], 'inner') \
    .join(df4, ['beneID', 'state'], 'inner') \
    .join(demo, ['beneID', 'state'], 'inner')


# Show the final DataFrame
print(df.count())
display(df)

# COMMAND ----------

# Keep rows with specific states
states_to_keep = ["AL", "ME", "MT", "VT", "WY", "IL"]
df_filtered = df.filter(col("state").isin(states_to_keep))
print(df_filtered.count())

# COMMAND ----------

inpatient = spark.table("dua_058828_spa240.paper1_stage2_inpatient_cost_15M")
inpatient = inpatient.withColumn("ffs_total", when(inpatient['CLM_TYPE_CD'] == '1', inpatient['MDCD_PD_AMT']).otherwise(None)) \
           .withColumn("mc_total", when(inpatient['CLM_TYPE_CD'] == '3', inpatient['LINE_MDCD_FFS_EQUIV_AMT']).otherwise(None))
inpatient = inpatient.fillna(0.0, subset=["ffs_total", "mc_total"])
inpatient = inpatient.select("beneID","state","CLM_ID","LINE_NUM","CLM_TYPE_CD","MDCD_PD_AMT","LINE_MDCD_FFS_EQUIV_AMT","ffs_total","mc_total","post")
inpatient.show(250)

#aggregate to claim 
inpatient.registerTempTable("missingTotal")
inpatient_spend = spark.sql('''
select beneID, CLM_ID, state, sum(mc_total) as mc_total, mean(ffs_total) as ffs_total
FROM missingTotal 
GROUP BY beneID, CLM_ID, state; 
''')

inpatient_spend = inpatient_spend.withColumn("total_cost", col("mc_total") + col("ffs_total"))
inpatient_final = inpatient_spend.select("beneID","state","mc_total","ffs_total","total_cost")
inpatient_final.show(100)

# COMMAND ----------

# inpatient = spark.table("dua_058828_spa240.paper1_stage2_long_term_care_cost")
# inpatient = inpatient.withColumn("ffs_total", when(inpatient['CLM_TYPE_CD'] == '1', inpatient['MDCD_PD_AMT']).otherwise(None)) \
#            .withColumn("mc_total", when(inpatient['CLM_TYPE_CD'] == '3', inpatient['LINE_MDCD_FFS_EQUIV_AMT']).otherwise(None))
# inpatient = inpatient.fillna(0.0, subset=["ffs_total", "mc_total"])
# inpatient = inpatient.select("beneID","state","CLM_ID","LINE_NUM","CLM_TYPE_CD","MDCD_PD_AMT","LINE_MDCD_FFS_EQUIV_AMT","ffs_total","mc_total","post")
# inpatient.show(250)

# #aggregate to claim 
# inpatient.registerTempTable("missingTotal")
# inpatient_spend = spark.sql('''
# select beneID, CLM_ID, state, sum(mc_total) as mc_total, mean(ffs_total) as ffs_total
# FROM missingTotal 
# GROUP BY beneID, CLM_ID, state; 
# ''')

# inpatient_spend = inpatient_spend.withColumn("total_cost", col("mc_total") + col("ffs_total"))
# ltc_final = inpatient_spend.select("beneID","state","mc_total","ffs_total","total_cost")
# ltc_final.show(100)

# COMMAND ----------

inpatient = spark.table("dua_058828_spa240.paper1_stage2_pharm_cost_15M")
inpatient = inpatient.withColumn("ffs_total", when(inpatient['CLM_TYPE_CD'] == '1', inpatient['MDCD_PD_AMT']).otherwise(None)) \
           .withColumn("mc_total", when(inpatient['CLM_TYPE_CD'] == '3', inpatient['LINE_MDCD_FFS_EQUIV_AMT']).otherwise(None))
inpatient = inpatient.fillna(0.0, subset=["ffs_total", "mc_total"])
inpatient = inpatient.select("beneID","state","CLM_ID","LINE_NUM","CLM_TYPE_CD","MDCD_PD_AMT","LINE_MDCD_FFS_EQUIV_AMT","ffs_total","mc_total","post")
inpatient.show(250)

#aggregate to claim 
inpatient.registerTempTable("missingTotal")
inpatient_spend = spark.sql('''
select beneID, CLM_ID, state, sum(mc_total) as mc_total, mean(ffs_total) as ffs_total
FROM missingTotal 
GROUP BY beneID, CLM_ID, state; 
''')

inpatient_spend = inpatient_spend.withColumn("total_cost", col("mc_total") + col("ffs_total"))
pharm_final = inpatient_spend.select("beneID","state","mc_total","ffs_total","total_cost")
pharm_final.show(100)

# COMMAND ----------

inpatient = spark.table("dua_058828_spa240.paper1_stage2_outpatient_cost_15M")
inpatient = inpatient.withColumn("ffs_total", when(inpatient['CLM_TYPE_CD'] == '1', inpatient['LINE_MDCD_PD_AMT']).otherwise(None)) \
           .withColumn("mc_total", when(inpatient['CLM_TYPE_CD'] == '3', inpatient['LINE_MDCD_FFS_EQUIV_AMT']).otherwise(None))
inpatient = inpatient.fillna(0.0, subset=["ffs_total", "mc_total"])
inpatient = inpatient.select("beneID","state","CLM_ID","LINE_NUM","CLM_TYPE_CD","LINE_MDCD_PD_AMT","LINE_MDCD_FFS_EQUIV_AMT","ffs_total","mc_total","post")
inpatient.show(25)

#aggregate to claim 
inpatient.registerTempTable("missingTotal")
inpatient_spend = spark.sql('''
select beneID, CLM_ID, state, sum(mc_total) as mc_total, sum(ffs_total) as ffs_total
FROM missingTotal 
GROUP BY beneID, CLM_ID, state; 
''')

inpatient_spend = inpatient_spend.withColumn("total_cost", col("mc_total") + col("ffs_total"))
outpatient_final = inpatient_spend.select("beneID","state","mc_total","ffs_total","total_cost")
outpatient_final.show(100)

# COMMAND ----------

# all_files = pharm_final.union(outpatient_final).union(inpatient_final)
# all_files.show(20)

all_files =inpatient_final

# COMMAND ----------

#aggregate to beneID-state level 

all_files.registerTempTable("all_spend")
total_spend = spark.sql('''
select beneID, state, sum(mc_total) as mc_total, sum(ffs_total) as ffs_total, sum(total_cost) as total_cost
FROM all_spend 
GROUP BY beneID, state; 
''')

total_spend.show()

# COMMAND ----------

final = df_filtered.join(total_spend, ['beneID', 'state'], 'left').fillna(0)
final.show(100)

# COMMAND ----------

final.write.saveAsTable("dua_058828_spa240.paper1_stage2_final_data_2M", mode='overwrite') 

# COMMAND ----------

final.groupBy("ageCat").count().show()
final.groupBy("disabled").count().show()

# COMMAND ----------

adults = final.filter((final.disabled == "no") & (~final.ageCat.isin(["under10", "10To17","missing"])))
kids=final.filter((final.disabled == "no") & (final.ageCat.isin(["under10", "10To17"])))
disabled = final.filter(final.disabled == "yes") 

# COMMAND ----------

disabled.groupBy("ageCat").count().show()
disabled.groupBy("disabled").count().show()

adults.groupBy("ageCat").count().show()
adults.groupBy("disabled").count().show()

kids.groupBy("ageCat").count().show()
kids.groupBy("disabled").count().show()

# COMMAND ----------

kids.write.saveAsTable("dua_058828_spa240.paper1_stage2_final_data_kids_2M", mode='overwrite') 
adults.write.saveAsTable("dua_058828_spa240.paper1_stage2_final_data_adults_2M", mode='overwrite') 
disabled.write.saveAsTable("dua_058828_spa240.paper1_stage2_final_data_disabled_2M", mode='overwrite') 

# COMMAND ----------

