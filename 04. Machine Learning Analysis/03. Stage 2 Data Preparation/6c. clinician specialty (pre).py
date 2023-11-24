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
outpat = outpat.select("beneID","state","CLM_ID","SRVC_BGN_DT","SRVC_PRVDR_NPI","SRVC_PRVDR_SPCLTY_CD","BLG_PRVDR_NPI","BLG_PRVDR_SPCLTY_CD","REV_CNTR_CD","POS_CD","LINE_PRCDR_CD")
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
outpat_selected = outpat_selected.drop("REV_CNTR_CD","POS_CD","LINE_PRCDR_CD","EDvisit")
 
# Show the result
outpat_selected.show(1000)

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Assuming you have a DataFrame named 'outpat_selected'

# Check for missing values in the 'SRVC_PRVDR_NPI' column
missing_srvc_prvdr_npi = outpat_selected.select(sum(col('SRVC_PRVDR_NPI').isNull().cast('integer'))).collect()[0][0]

# Check for missing values in the 'SRVC_PRVDR_SPCLTY_CD' column
missing_srvc_prvdr_spclty_cd = outpat_selected.select(sum(col('SRVC_PRVDR_SPCLTY_CD').isNull().cast('integer'))).collect()[0][0]

# Check for missing values in the 'BLG_PRVDR_NPI' column
missing_blg_prvdr_npi = outpat_selected.select(sum(col('BLG_PRVDR_NPI').isNull().cast('integer'))).collect()[0][0]

# Check for missing values in the 'BLG_PRVDR_SPCLTY_CD' column
missing_blg_prvdr_spclty_cd = outpat_selected.select(sum(col('BLG_PRVDR_SPCLTY_CD').isNull().cast('integer'))).collect()[0][0]

# Print the results
print('Missing SRVC_PRVDR_NPI:', missing_srvc_prvdr_npi)
print('Missing SRVC_PRVDR_SPCLTY_CD:', missing_srvc_prvdr_spclty_cd)
print('Missing BLG_PRVDR_NPI:', missing_blg_prvdr_npi)
print('Missing BLG_PRVDR_SPCLTY_CD:', missing_blg_prvdr_spclty_cd)

#total lines 257,731,014
#36,929,830
#84,731,160
#21,482,325
#64,506,633

# COMMAND ----------

from pyspark.sql.functions import col, sum, when

# Assuming you have a DataFrame named 'outpat_selected'

# Calculate the count of non-null occurrences in either 'SRVC_PRVDR_SPCLTY_CD' or 'BLG_PRVDR_SPCLTY_CD'
non_null_count = outpat_selected.select(sum(when((col('SRVC_PRVDR_SPCLTY_CD').isNotNull()) | (col('BLG_PRVDR_SPCLTY_CD').isNotNull()), 1).otherwise(0))).collect()[0][0]

# Calculate the percentage of non-null occurrences
total_records = outpat_selected.count()
non_null_percentage = (non_null_count / total_records) * 100

# Print the result
print('Non-null occurrence count:', non_null_count)
print('Non-null occurrence percentage:', non_null_percentage)

# COMMAND ----------

from pyspark.sql.functions import col

# Assuming you have a DataFrame named 'outpat_selected'

# Filter the DataFrame to exclude rows where both 'SRVC_PRVDR_SPCLTY_CD' and 'BLG_PRVDR_SPCLTY_CD' are null
filtered_df = outpat_selected.filter(~(col('SRVC_PRVDR_SPCLTY_CD').isNull() & col('BLG_PRVDR_SPCLTY_CD').isNull()))

# Create the 'specialty_code' column based on conditions
filtered_df = filtered_df.withColumn('specialty_code', when(col('SRVC_PRVDR_SPCLTY_CD').isNotNull(), col('SRVC_PRVDR_SPCLTY_CD')).otherwise(col('BLG_PRVDR_SPCLTY_CD')))

# Print the filtered DataFrame
filtered_df.show()

# COMMAND ----------

filtered_df.registerTempTable("connections")

specialty = spark.sql('''

SELECT DISTINCT beneID, state, SRVC_BGN_DT, specialty_code

FROM connections;
''')

specialty = specialty.filter(col("specialty_code").isNotNull())
specialty.show(200)

# COMMAND ----------

#print((diagnosis_with_ccsr.count(), len(diagnosis_with_ccsr.columns)))
specialty = specialty.select("beneID","state","specialty_code")
specialty.show(500)

# COMMAND ----------

# Pivot the DataFrame to create indicator columns for each unique ccsr value
pivoted_df = specialty.groupBy("beneID", "state").pivot("specialty_code").agg({"specialty_code": "count"}).fillna(0)

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
specialty_predictor = member.join(pivoted_df, on=['beneID','state'], how='left').fillna(0)
print((specialty_predictor.count(), len(specialty_predictor.columns)))

# COMMAND ----------

display(specialty_predictor)

# COMMAND ----------

specialty_predictor.write.saveAsTable("dua_058828_spa240.paper1_clinician_specialty_predictors", mode='overwrite') 

# COMMAND ----------

