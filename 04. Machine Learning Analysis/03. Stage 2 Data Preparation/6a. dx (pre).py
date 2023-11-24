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

#dx

#outpat
outpat = spark.table("dua_058828_spa240.paper1_stage2_outpatient_12month")
print((outpat.count(), len(outpat.columns)))
outpat = outpat.filter(outpat.pre==1)
print((outpat.count(), len(outpat.columns)))
outpat = outpat.select("beneID","state","CLM_ID","SRVC_BGN_DT","DGNS_CD_1")
print(outpat.printSchema())

# COMMAND ----------

#inpat

inpatient = spark.table("dua_058828_spa240.paper1_stage2_inpatient_12month")
print((inpatient.count(), len(inpatient.columns)))
inpatient = inpatient.filter(inpatient.pre==1)
print((inpatient.count(), len(inpatient.columns)))
inpatient = inpatient.select("beneID","state","CLM_ID","SRVC_BGN_DT","DGNS_CD_1")
print(inpatient.printSchema())

# COMMAND ----------

#long term care

longterm = spark.table("dua_058828_spa240.paper1_stage2_longterm_12month")
print((longterm.count(), len(longterm.columns)))
longterm = longterm.filter(longterm.pre==1)
print((longterm.count(), len(longterm.columns)))
longterm = longterm.select("beneID","state","CLM_ID","SRVC_BGN_DT","DGNS_CD_1")
print(longterm.printSchema())

# COMMAND ----------

diagnosis = outpat.union(inpatient).union(longterm)

diagnosis.registerTempTable("connections")

diagnosis1 = spark.sql('''

SELECT DISTINCT beneID, state, SRVC_BGN_DT, DGNS_CD_1

FROM connections;
''')

diagnosis2 = diagnosis1.filter(col("DGNS_CD_1").isNotNull())
diagnosis2.show(200)

# COMMAND ----------

import pandas as pd
ccsr_df = pd.read_csv("/dbfs/mnt/dua/dua_058828/SPA240/files/ccsrCategory.csv")
ccsr_df["diagnosisOne"] = ccsr_df["diagnosisOne"].str.strip("'")
ccsr_df["ccsrCatOne"] = ccsr_df["ccsrCatOne"].str.strip("'")
#ccsr_df.head()
spark_ccsr = spark.createDataFrame(ccsr_df)
spark_ccsr = spark_ccsr.withColumnRenamed("diagnosisOne", "DGNS_CD_1")
spark_ccsr = spark_ccsr.withColumnRenamed("ccsrCatOne", "ccsr")
spark_ccsr = spark_ccsr.select("DGNS_CD_1","ccsr")
spark_ccsr.show()

# COMMAND ----------

spark_ccsr_first_three = spark_ccsr.withColumn('first_three', substring(col('DGNS_CD_1'), 1, 3))
spark_ccsr_first_three.registerTempTable("connections")

spark_ccsr_first_three = spark.sql('''

SELECT DISTINCT first_three, max(ccsr) as ccsr

FROM connections
GROUP BY first_three;
''')

spark_ccsr_first_three.show(200)

# COMMAND ----------

from pyspark.sql.functions import col, substring

# Join on the full DGNS_CD_1
diagnosis_with_ccsr = diagnosis2.join(
    spark_ccsr,
    on='DGNS_CD_1',
    how='left'
)

print(diagnosis_with_ccsr.count())
missing_count = diagnosis_with_ccsr.select("ccsr").where(col("ccsr").isNull()).count()
print(missing_count)

# Filter out matched rows
matched_rows = diagnosis_with_ccsr.filter(diagnosis_with_ccsr['ccsr'].isNotNull())

# Filter out unmatched rows
unmatched_rows = diagnosis_with_ccsr.filter(diagnosis_with_ccsr['ccsr'].isNull())
unmatched_rows = unmatched_rows.drop('ccsr')
unmatched_rows = unmatched_rows.withColumn('first_three', substring(col('DGNS_CD_1'), 1, 3))

# matched_rows.show(10)
# unmatched_rows.show(10)

# Join on the first 3 characters of DGNS_CD_1 for unmatched rows
unmatched_rows = unmatched_rows.join(spark_ccsr_first_three, on=(unmatched_rows['first_three'] == spark_ccsr_first_three['first_three']), how='left')
unmatched_rows = unmatched_rows.drop("first_three")

#matched_rows.show(10)
#unmatched_rows.show(10)

# # Union the matched and unmatched rows
diagnosis_with_ccsr = matched_rows.union(unmatched_rows)

print(diagnosis_with_ccsr.count())
missing_count = diagnosis_with_ccsr.select("ccsr").where(col("ccsr").isNull()).count()
print(missing_count)

#first iteration (perfect match)
#5,329,623 missing CCSR
#96,481,445

#second iteration (perfect match + fuzzy match)
#4,720,483 missing CCSR
#96,481,445

# COMMAND ----------

#print((diagnosis_with_ccsr.count(), len(diagnosis_with_ccsr.columns)))
diagnosis_with_ccsr = diagnosis_with_ccsr.select("beneID","state","ccsr")
diagnosis_with_ccsr.show(500)

# COMMAND ----------

# Pivot the DataFrame to create indicator columns for each unique ccsr value
pivoted_df = diagnosis_with_ccsr.groupBy("beneID", "state").pivot("ccsr").agg({"ccsr": "count"}).fillna(0)

# Show the pivoted DataFrame
pivoted_df.show()

# COMMAND ----------

print((pivoted_df.count(), len(pivoted_df.columns)))
num_distinct= pivoted_df.select(["beneID","state"]).distinct().count()
print(num_distinct)

# COMMAND ----------

print(pivoted_df.printSchema())

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf

member = spark.table("dua_058828_spa240.demographic_stage2")
member = member.select("beneID", "state")
print((member.count(), len(member.columns)))
print(member.printSchema())

# # Left join 'df' with 'ed' based on the 'DGNS_CD_1' column
dx_predictor = member.join(pivoted_df, on=['beneID','state'], how='left').fillna(0)
print((dx_predictor.count(), len(dx_predictor.columns)))

# COMMAND ----------

display(dx_predictor)

# COMMAND ----------

dx_predictor.write.saveAsTable("dua_058828_spa240.paper1_dx_predictors", mode='overwrite') 

# COMMAND ----------

