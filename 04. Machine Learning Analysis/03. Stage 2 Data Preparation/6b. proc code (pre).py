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

#proc code

#outpat only -- other files do not have a proc code
outpat = spark.table("dua_058828_spa240.paper1_stage2_outpatient_12month")
print((outpat.count(), len(outpat.columns)))
outpat = outpat.filter(outpat.pre==1)
print((outpat.count(), len(outpat.columns)))
outpat = outpat.select("beneID","state","CLM_ID","SRVC_BGN_DT","LINE_PRCDR_CD","REV_CNTR_CD","POS_CD")
print(outpat.printSchema())

# COMMAND ----------

#remove ED visits because measured elsewhere
# Define the conditions for the "EDvisit" binary indicator

edvisit_conditions = (
    outpat["REV_CNTR_CD"].isin(['0450', '0451', '0452', '0453', '0454', '0456', '0457', '0458', '0459', '0981']) |
    outpat["POS_CD"].isin([23]) |
    outpat["LINE_PRCDR_CD"].isin(['99281', '99282', '99283', '99284', '99285'])
)
 
# Create the "EDvisit" binary indicator based on the conditions
outpat = outpat.withColumn("EDvisit", when(edvisit_conditions, 1).otherwise(0))
 
# Filter out rows where "EDvisit" is not equal to 1
outpat_selected = outpat.filter(outpat["EDvisit"] == 0)
 
# Show the result
outpat_selected.show(1000)

# COMMAND ----------

outpat_selected.registerTempTable("connections")

proc_code = spark.sql('''

SELECT DISTINCT beneID, state, SRVC_BGN_DT, LINE_PRCDR_CD

FROM connections;
''')

proc_code1 = proc_code.filter(col("LINE_PRCDR_CD").isNotNull())
proc_code1.show(200)

# COMMAND ----------

import pandas as pd
betos_df = pd.read_csv("/dbfs/mnt/dua/dua_058828/SPA240/files/rbcsCrossWalk.csv")
#betos_df.head()
betos_df = spark.createDataFrame(betos_df)
betos_df = betos_df.withColumnRenamed("procedureCode", "LINE_PRCDR_CD")
betos_df = betos_df.withColumnRenamed("rbcsID", "betos")
betos_df = betos_df.select("LINE_PRCDR_CD","betos")
betos_df.show()

# COMMAND ----------

spark_betos_first_three = betos_df.withColumn('first_three', substring(col('LINE_PRCDR_CD'), 1, 3))
spark_betos_first_three.registerTempTable("connections")

spark_betos_first_three = spark.sql('''

SELECT DISTINCT first_three, max(betos) as betos

FROM connections
GROUP BY first_three;
''')

spark_betos_first_three.show(200)

# COMMAND ----------

from pyspark.sql.functions import col, substring

# Join on the full DGNS_CD_1
proc_code_with_betos = proc_code1.join(
    betos_df,
    on='LINE_PRCDR_CD',
    how='left'
)

print(proc_code_with_betos.count())
missing_count = proc_code_with_betos.select("betos").where(col("betos").isNull()).count()
print(missing_count)

# Filter out matched rows
matched_rows = proc_code_with_betos.filter(proc_code_with_betos['betos'].isNotNull())

# Filter out unmatched rows
unmatched_rows = proc_code_with_betos.filter(proc_code_with_betos['betos'].isNull())
unmatched_rows = unmatched_rows.drop('betos')
unmatched_rows = unmatched_rows.withColumn('first_three', substring(col('LINE_PRCDR_CD'), 1, 3))

# matched_rows.show(10)
# unmatched_rows.show(10)

# Join on the first 3 characters of LINE_PRCDR_CD for unmatched rows
unmatched_rows = unmatched_rows.join(spark_betos_first_three, on=(unmatched_rows['first_three'] == spark_betos_first_three['first_three']), how='left')
unmatched_rows = unmatched_rows.drop("first_three")

#matched_rows.show(10)
#unmatched_rows.show(10)

# # Union the matched and unmatched rows
proc_code_with_betos = matched_rows.union(unmatched_rows)

print(proc_code_with_betos.count())
missing_count = proc_code_with_betos.select("betos").where(col("betos").isNull()).count()
print(missing_count)

#first iteration (perfect match)
#32,753,313 missing BETOS
#196,501,632

#second iteration (perfect match + fuzzy match)
#8,363,352 missing BETOS
#196,501,632

# COMMAND ----------

#print((diagnosis_with_ccsr.count(), len(diagnosis_with_ccsr.columns)))
proc_code_with_betos = proc_code_with_betos.select("beneID","state","betos")
proc_code_with_betos.show(500)

# COMMAND ----------

# Pivot the DataFrame to create indicator columns for each unique ccsr value
pivoted_df = proc_code_with_betos.groupBy("beneID", "state").pivot("betos").agg({"betos": "count"}).fillna(0)

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
proc_code_predictor = member.join(pivoted_df, on=['beneID','state'], how='left').fillna(0)
print((proc_code_predictor.count(), len(proc_code_predictor.columns)))

# COMMAND ----------

display(proc_code_predictor)

# COMMAND ----------

proc_code_predictor.write.saveAsTable("dua_058828_spa240.paper1_proc_code_predictors", mode='overwrite') 

# COMMAND ----------

