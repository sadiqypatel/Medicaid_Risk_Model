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

pharm_pre.groupby('keep').count().orderBy(desc('count'),'keep').show()
pharm_pre.groupby('pre').count().orderBy(desc('count'),'pre').show()
pharm_pre.groupby('post').count().orderBy(desc('count'),'post').show()

# COMMAND ----------

print((pharm_pre.count(), len(pharm_pre.columns)))
print(pharm_pre.printSchema())

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_add
from pyspark.sql.functions import expr

pharm_pre_selected = pharm_pre.select("beneID", "state", "CLM_ID", "RX_FILL_DT", "NDC", "PRSCRBNG_PRVDR_NPI", "DAYS_SUPPLY", "NEW_RX_REFILL_NUM" ,"BRND_GNRC_CD")
pharm_pre_selected = pharm_pre_selected.withColumn("generic", F.when(pharm_pre_selected["BRND_GNRC_CD"] == 1, 1).otherwise(0))
pharm_pre_selected = pharm_pre_selected.withColumn("number_of_fills", lit(1))
# Cast the "DAYS_SUPPLY" column to integer
pharm_pre_selected = pharm_pre_selected.withColumn("DAYS_SUPPLY", pharm_pre_selected["DAYS_SUPPLY"].cast("integer"))

# Add a new column "SUM_DATE" representing the sum of "RX_FILL_DT" and "DAYS_SUPPLY"
pharm_pre_selected = pharm_pre_selected.withColumn("SUM_DATE", expr("date_add(RX_FILL_DT, DAYS_SUPPLY)"))
pharm_pre_selected.show(250)

# COMMAND ----------

pharm_pre_selected.registerTempTable("connections")

pharm_pre_selected_agg = spark.sql('''

SELECT beneID, state, ndc, sum(number_of_fills) as number_of_fills, min(RX_FILL_DT) as first_rx_fill_date,  max(SUM_DATE) as last_rx_fill_date, sum(DAYS_SUPPLY) as days_supply,
sum(generic) as generic

FROM connections
GROUP BY beneID, state, ndc;
''')

pharm_pre_selected_agg.show(200)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff

pharm_pre_selected_agg = pharm_pre_selected_agg.withColumn("date_difference", datediff(col("last_rx_fill_date"), col("first_rx_fill_date")))
pharm_pre_selected_agg = pharm_pre_selected_agg.withColumn("number_of_unique_med", lit(1))
pharm_pre_selected_agg.show(250)

# COMMAND ----------

pharm_pre_selected_agg.registerTempTable("connections")

pharm_final_1 = spark.sql('''

SELECT beneID, state, sum(number_of_fills) as number_of_fills, sum(number_of_unique_med) as number_of_unique_med, sum(generic) as generic, sum(days_supply) as days_supply, sum(date_difference) as date_difference 

FROM connections
GROUP BY beneID, state;
''')

pharm_final_1.show(200)

# COMMAND ----------

from pyspark.sql.functions import col, round

# Calculate the percentage of generic drugs
pharm_final_1 = pharm_final_1.withColumn("percent_generic", round(col("generic") / col("number_of_fills"), 4))

# Calculate the percentage of medication adherence
pharm_final_1 = pharm_final_1.withColumn("percent_med_adherence", round(col("days_supply") / col("date_difference"), 4))

# Show the updated DataFrame
pharm_final_1.show()

# COMMAND ----------

pharm_final_first = pharm_final_1.select('beneID','state','number_of_fills','number_of_unique_med','percent_generic', 'percent_med_adherence')
pharm_final_first.show()

# COMMAND ----------

pharm = spark.table("dua_058828_spa240.paper1_stage2_pharm_12month")
print(pharm.count())
pharm_pre_slope = pharm.filter(pharm.pre==1)
print(pharm_pre_slope.count())
pharm_pre_slope = pharm_pre_slope.withColumn("number_of_fills", lit(1))

pharm_pre_slope.groupby('keep').count().orderBy(desc('count'),'keep').show()
pharm_pre_slope.groupby('pre').count().orderBy(desc('count'),'pre').show()
pharm_pre_slope.groupby('post').count().orderBy(desc('count'),'post').show()

# COMMAND ----------

print(pharm_pre_slope.printSchema())

# COMMAND ----------

# Update the columns based on conditions
updated_df = pharm_pre_slope.withColumn("pharm_month1", when(col("service_month") == col("pre_start"), col("number_of_fills")).otherwise(0))
updated_df = updated_df.withColumn("pharm_month2", when(col("service_month") == col("pre_start") +1, col("number_of_fills")).otherwise(0))
updated_df = updated_df.withColumn("pharm_month3", when(col("service_month") == col("pre_start") + 2, col("number_of_fills")).otherwise(0))
updated_df = updated_df.withColumn("pharm_month4", when(col("service_month") == col("pre_start") + 3, col("number_of_fills")).otherwise(0))
updated_df = updated_df.withColumn("pharm_month5", when(col("service_month") == col("pre_start") + 4, col("number_of_fills")).otherwise(0))
updated_df = updated_df.withColumn("pharm_month6", when(col("service_month") == col("pre_start") + 5, col("number_of_fills")).otherwise(0))

# COMMAND ----------

# Select the desired columns
selected = updated_df.select("beneID", "state", "RX_FILL_DT", "service_month", "pre_start", "pre_end", "pharm_month1", "pharm_month2", "pharm_month3", "pharm_month4", "pharm_month5", "pharm_month6")

# Show the final result
selected.show(500)

# COMMAND ----------

selected.registerTempTable("connections")

pharm_pre_slope = spark.sql('''

SELECT beneID, state, sum(pharm_month1) as pharm_month1, sum(pharm_month2) as pharm_month2, sum(pharm_month3) as pharm_month3, sum(pharm_month4) as pharm_month4, sum(pharm_month5) as pharm_month5,
sum(pharm_month6) as pharm_month6 

FROM connections
GROUP BY beneID, state;
''')

pharm_pre_slope.show(200)

# COMMAND ----------

from pyspark.sql.functions import array, col

# Assuming you have the DataFrame "acute_care_slope_agg"
column_names = ["pharm_month1", "pharm_month2", "pharm_month3", "pharm_month4", "pharm_month5", "pharm_month6"]

# Create a new column with a vector of values
test = pharm_pre_slope.withColumn("y_pharm", array(*[col(column) for column in column_names]))
test = test.withColumn("x", array(*[lit(i) for i in range(1, 7)]))

test = test.select("beneID", "state", "x", "y_pharm")

test.show()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import numpy as np

# Define a UDF to calculate the slope
def get_slope_func(x, y, order=1):
    coeffs = np.polyfit(x, y, order)
    slope = coeffs[-2]
    return float(slope)

# Register the UDF
get_slope = F.udf(get_slope_func, DoubleType())

# Calculate the slope for each row using linear fit (order=1)
test = test.withColumn("pharm_slope", get_slope(F.col("x"), F.col("y_pharm"), F.lit(1)))
# Round the slope values to 6 decimals
test = test.withColumn("pharm_slope", F.round(F.col("pharm_slope"), 6))

# Show the results
test.show(500)

# COMMAND ----------

# # Left join 'df' with 'ed' based on the 'DGNS_CD_1' column

member = spark.table("dua_058828_spa240.demographic_stage2")
member = member.select("beneID", "state")
print((member.count(), len(member.columns)))
print(member.printSchema())

from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf

# # Left join 'df' with 'ed' based on the 'DGNS_CD_1' column
pharm_predictor = member.join(test, on=['beneID','state'], how='left').fillna(0)
pharm_predictor = pharm_predictor.join(pharm_final_first, on=['beneID','state'], how='left').fillna(0)

pharm_predictor.show(1000)

# COMMAND ----------

pharm_predictor.write.saveAsTable("dua_058828_spa240.paper1_pharm_predictors", mode='overwrite') 

# COMMAND ----------

