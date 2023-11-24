# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import numpy as np

# COMMAND ----------

#inpatient claims
inpatient = spark.table("dua_058828_spa240.paper1_stage2_outpatient_cost")
print((inpatient.count(), len(inpatient.columns)))

# Count the number of distinct values in a column
distinct_count = inpatient.select(col("state")).distinct().count()

# Display the result
print("Distinct count:", distinct_count)
#print(inpatient.printSchema())

# COMMAND ----------

#create indicator for notNull
inpatient_missing = inpatient.withColumn("missingFFS", when((col("CLM_TYPE_CD") == "1") & (col("LINE_MDCD_PD_AMT").isNull()), lit(1)).otherwise(lit(0)))
inpatient_missing = inpatient_missing.withColumn("missingMC", when((col("CLM_TYPE_CD") == "3") & (col("LINE_MDCD_FFS_EQUIV_AMT").isNull()), lit(1)).otherwise(lit(0)))

inpatient_missing = inpatient_missing.withColumn("ffs_claim", when((col("CLM_TYPE_CD") == "1"), lit(1)).otherwise(lit(0)))
inpatient_missing = inpatient_missing.withColumn("mc_claim", when((col("CLM_TYPE_CD") == "3"), lit(1)).otherwise(lit(0)))
                                                           
# selected_columns = ["beneID", "state", "CLM_ID", "LINE_NUM", "CLM_TYPE_CD", "LINE_MDCD_FFS_EQUIV_AMT", "MDCD_PD_AMT", "MDCD_COPAY_AMT", "MDCR_PD_AMT", "missingFFS", "missingMC", "ffs_claim","mc_claim"]
# inpatient_missing = inpatient_missing.select(*selected_columns)
# inpatient_missing.show(1000)

inpatient_missing.registerTempTable("missingTable")
missingSpend = spark.sql('''
select distinct beneID, state, CLM_ID, max(missingFFS) as missingFFS, max(missingMC) as missingMC, max(ffs_claim) as ffs_claim, max(mc_claim) as mc_claim
FROM missingTable 
GROUP BY beneID, state, CLM_ID; 
''')

missingSpend = missingSpend.withColumn("totalClaims", lit(1))

#aggregate to total 
missingSpend.registerTempTable("missingTotal")
missingSpendState = spark.sql('''
select state, sum(missingFFS) as missingFFS, sum(missingMC) as missingMC, sum(ffs_claim) as ffs_claim, sum(mc_claim) as mc_claim, sum(totalClaims) as total
FROM missingTotal 
GROUP BY state; 
''')

missingSpendState.show(n=missingSpendState.count(), truncate=False)

# COMMAND ----------

from pyspark.sql.functions import expr

df_new = inpatient.withColumn("ffs_total", when(inpatient['CLM_TYPE_CD'] == '1', inpatient['LINE_MDCD_PD_AMT']).otherwise(None)) \
           .withColumn("mc_total", when(inpatient['CLM_TYPE_CD'] == '3', inpatient['LINE_MDCD_FFS_EQUIV_AMT']).otherwise(None))

df_new = df_new.fillna(0.0, subset=["ffs_total", "mc_total"])

df_new = df_new.withColumn("ffs_claim", when(inpatient['CLM_TYPE_CD'] == '1', lit(1)).otherwise(lit(0))) \
           .withColumn("mc_claim", when(inpatient['CLM_TYPE_CD'] == '3', lit(1)).otherwise(lit(0)))

selected_columns = ["beneID", "state", "CLM_ID", "LINE_NUM", "CLM_TYPE_CD", "LINE_MDCD_FFS_EQUIV_AMT", "LINE_MDCD_PD_AMT", "ffs_total", "mc_total", "ffs_claim", "mc_claim"]
df_selected = df_new.select(*selected_columns)
df_selected.show(1000)

# COMMAND ----------

#aggregate to total 
df_selected.registerTempTable("missingTotal")
spend_claim = spark.sql('''
select beneID, CLM_ID, state, sum(mc_total) as mc_total,  sum(ffs_total) as ffs_total, sum(ffs_claim) as ffs_claim, sum(mc_claim) as mc_claim
FROM missingTotal 
GROUP BY beneID, CLM_ID, state; 
''')

#aggregate to state 
spend_claim.registerTempTable("missingTotal")
spend_state = spark.sql('''
select state, sum(mc_total) as mc_total, sum(ffs_total) as ffs_total, sum(ffs_claim) as ffs_claim_lines, sum(mc_claim) as mc_claim_lines
FROM missingTotal 
GROUP BY state; 
''')

spend_state.show(n=spend_state.count(), truncate=False)

# COMMAND ----------

