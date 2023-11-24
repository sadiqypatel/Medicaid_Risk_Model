# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

pharm = spark.table("dua_058828_spa240.paper1_stage2_pharm_12month")
print(pharm.count())
pharm_pre = pharm.filter(pharm.pre==1)
print(pharm_pre.count())
pharm_pre = pharm_pre.select("beneID", "state", "RX_FILL_DT", "NDC")

# COMMAND ----------

print((pharm_pre.count(), len(pharm_pre.columns)))
print(pharm_pre.printSchema())

# COMMAND ----------

pharm_pre.registerTempTable("connections")

pharm_pre = spark.sql('''

SELECT DISTINCT beneID, state, RX_FILL_DT, NDC

FROM connections;
''')

pharm_pre = pharm_pre.filter(col("NDC").isNotNull())
pharm_pre.show(200)

# COMMAND ----------

import pandas as pd
rx_df = pd.read_csv("/dbfs/mnt/dua/dua_058828/SPA240/files/rxfile.csv")
#betos_df.head()
rx_df = spark.createDataFrame(rx_df)
rx_df = rx_df.withColumnRenamed("ndcNum", "ndc")
rx_df.show()

# COMMAND ----------

from pyspark.sql.functions import col, substring

rx_with_agg = pharm_pre.join(
    rx_df,
    on='ndc',
    how='left'
)

print(rx_with_agg.count())
missing_count = rx_with_agg.select("rxDcClassCode").where(col("rxDcClassCode").isNull()).count()
print(missing_count)

#total: 75,338,376
#missing: 8,297,318

# COMMAND ----------

#print((diagnosis_with_ccsr.count(), len(diagnosis_with_ccsr.columns)))
rx_with_agg = rx_with_agg.select("beneID","state","rxDcClassCode")
rx_with_agg.show(500)

# COMMAND ----------

# Pivot the DataFrame to create indicator columns for each unique ccsr value
pivoted_df = rx_with_agg.groupBy("beneID", "state").pivot("rxDcClassCode").agg({"rxDcClassCode": "count"}).fillna(0)

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
rx_predictors = member.join(pivoted_df, on=['beneID','state'], how='left').fillna(0)
print((rx_predictors.count(), len(rx_predictors.columns)))

# COMMAND ----------

display(rx_predictors)

# COMMAND ----------

rx_predictors.write.saveAsTable("dua_058828_spa240.paper1_rx_predictors", mode='overwrite') 

# COMMAND ----------

