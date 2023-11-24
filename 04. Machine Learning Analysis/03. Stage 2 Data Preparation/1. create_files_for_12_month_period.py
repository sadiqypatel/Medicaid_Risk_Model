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

stage1 = spark.table("dua_058828_spa240.stage1_sample")
df = spark.table("dua_058828_spa240.stage2_sample")

print((stage1.count(), len(stage1.columns)))
print((df.count(), len(df.columns)))
#print(df.printSchema())

# COMMAND ----------

df = df.withColumn("cov2016yes", when(col("2016months") > 0, 1).otherwise(0))
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

df.show(10)

# COMMAND ----------

df.write.saveAsTable("dua_058828_spa240.demographic_stage2", mode="overwrite")

# COMMAND ----------

df = spark.table("dua_058828_spa240.demographic_stage2")
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf

# Count the number of distinct values in a column
df.select(col("enrollMonth")).distinct().show()
df.select(col("enrollYear")).distinct().show()
df.select(col("first")).distinct().show()

# COMMAND ----------

# Assign values 1-24 based on enrollMonth and enrollYear

df = df.select("beneID", "state", "enrollYear","enrollMonth")

df = df.withColumn("MonthValue", (col("enrollYear") - 2017) * 12 + 
                 when(col("enrollMonth") == "Jan", 1)
                .when(col("enrollMonth") == "Feb", 2)
                .when(col("enrollMonth") == "Mar", 3)
                .when(col("enrollMonth") == "Apr", 4)
                .when(col("enrollMonth") == "May", 5)
                .when(col("enrollMonth") == "Jun", 6)
                .when(col("enrollMonth") == "Jul", 7)
                .when(col("enrollMonth") == "Aug", 8)
                .when(col("enrollMonth") == "Sep", 9)
                .when(col("enrollMonth") == "Oct", 10)
                .when(col("enrollMonth") == "Nov", 11)
                .when(col("enrollMonth") == "Dec", 12)
                .otherwise(None))

# Calculate start month, end month, pre_start, pre_end, post_start, and post_end
df = df.withColumn("start_month", col("MonthValue") + 1)
df = df.withColumn("end_month", col("MonthValue") + 12)
df = df.withColumn("pre_start", col("MonthValue") + 1)
df = df.withColumn("pre_end", col("MonthValue") + 6)
df = df.withColumn("post_start", col("MonthValue") + 7)
df = df.withColumn("post_end", col("MonthValue") + 12)

# Show the final result
df.show(500)

# COMMAND ----------

outpat2017 = spark.table("dua_058828_spa240.otherservices_final2017")
outpat2018 = spark.table("dua_058828_spa240.otherservices_final2018")
outpat2019 = spark.table("dua_058828_spa240.otherservices_final2019")

# Union the three dataframes
outpat = outpat2017.union(outpat2018).union(outpat2019)

print(outpat.printSchema())

# COMMAND ----------

from pyspark.sql.functions import month, year, col, lit

#print(outpat.count())
outpat_merge = outpat.join(df, on=['beneID','state'], how='inner')
#print(outpat_merge.count())

# Map "SRVC_BGN_DT" to a "service_month" value from 1 to 36
outpat_merge = outpat_merge.withColumn("service_month", 
                                       (year(col("SRVC_BGN_DT")) - 2017) * 12 + month(col("SRVC_BGN_DT")))

# Create "keep" column with a value of 0 or 1 based on "start_month" and "end_month" comparison
outpat_merge = outpat_merge.withColumn("keep", 
                                       when((col("start_month") <= col("service_month")) & 
                                            (col("service_month") <= col("end_month")), lit(1))
                                       .otherwise(lit(0)))

# Create the "pre" indicator
outpat_merge = outpat_merge.withColumn("pre", 
                                       when((col("pre_start") <= col("service_month")) & 
                                            (col("service_month") <= col("pre_end")), 1)
                                       .otherwise(0))

# Create the "post" indicator
outpat_merge = outpat_merge.withColumn("post", 
                                       when((col("post_start") <= col("service_month")) & 
                                            (col("service_month") <= col("post_end")), 1)
                                       .otherwise(0))

# COMMAND ----------

# Select the desired columns
selected = outpat_merge.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

# Show the final result
selected.show(500)

# COMMAND ----------

print(outpat_merge.count())
filtered_df = outpat_merge.filter(outpat_merge["keep"] == 1)
print(filtered_df.count())

# Select the desired columns
selected = filtered_df.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

selected.groupby('keep').count().orderBy(desc('count'),'keep').show()
selected.groupby('pre').count().orderBy(desc('count'),'pre').show()
selected.groupby('post').count().orderBy(desc('count'),'post').show()

# Show the final result
selected.show(500)

# COMMAND ----------

filtered_df.write.saveAsTable("dua_058828_spa240.paper1_stage2_outpatient_12month", mode='overwrite') 

# COMMAND ----------

inpat2017 = spark.table("dua_058828_spa240.inpatient_final2017")
inpat2018 = spark.table("dua_058828_spa240.inpatient_final2018")
inpat2019 = spark.table("dua_058828_spa240.inpatient_final2019")

# Union the three dataframes
inpat = inpat2017.union(inpat2018).union(inpat2019)

print(inpat.printSchema())

# COMMAND ----------

from pyspark.sql.functions import month, year, col, lit

#print(outpat.count())
inpat_merge = inpat.join(df, on=['beneID','state'], how='inner')
#print(outpat_merge.count())

# Map "SRVC_BGN_DT" to a "service_month" value from 1 to 36
inpat_merge = inpat_merge.withColumn("service_month", 
                                       (year(col("SRVC_BGN_DT")) - 2017) * 12 + month(col("SRVC_BGN_DT")))

# Create "keep" column with a value of 0 or 1 based on "start_month" and "end_month" comparison
inpat_merge = inpat_merge.withColumn("keep", 
                                       when((col("start_month") <= col("service_month")) & 
                                            (col("service_month") <= col("end_month")), lit(1))
                                       .otherwise(lit(0)))

# Create the "pre" indicator
inpat_merge = inpat_merge.withColumn("pre", 
                                       when((col("pre_start") <= col("service_month")) & 
                                            (col("service_month") <= col("pre_end")), 1)
                                       .otherwise(0))

# Create the "post" indicator
inpat_merge = inpat_merge.withColumn("post", 
                                       when((col("post_start") <= col("service_month")) & 
                                            (col("service_month") <= col("post_end")), 1)
                                       .otherwise(0))

# COMMAND ----------

# Select the desired columns
selected = inpat_merge.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

# Show the final result
selected.show(500)

# COMMAND ----------

print(inpat_merge.count())
filtered_df = inpat_merge.filter(inpat_merge["keep"] == 1)
print(filtered_df.count())

# Select the desired columns
selected = filtered_df.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

selected.groupby('keep').count().orderBy(desc('count'),'keep').show()
selected.groupby('pre').count().orderBy(desc('count'),'pre').show()
selected.groupby('post').count().orderBy(desc('count'),'post').show()

# Show the final result
selected.show(500)

# COMMAND ----------

filtered_df.write.saveAsTable("dua_058828_spa240.paper1_stage2_inpatient_12month", mode='overwrite') 

# COMMAND ----------

longterm2017 = spark.table("dua_058828_spa240.longterm_final2017")
longterm2018 = spark.table("dua_058828_spa240.longterm_final2018")
longterm2019 = spark.table("dua_058828_spa240.longterm_final2019")

# Union the three dataframes
longterm = longterm2017.union(longterm2018).union(longterm2019)

print(longterm.printSchema())

# COMMAND ----------

from pyspark.sql.functions import month, year, col, lit

#print(outpat.count())
longterm_merge = longterm.join(df, on=['beneID','state'], how='inner')
#print(outpat_merge.count())

# Map "SRVC_BGN_DT" to a "service_month" value from 1 to 36
longterm_merge = longterm_merge.withColumn("service_month", 
                                       (year(col("SRVC_BGN_DT")) - 2017) * 12 + month(col("SRVC_BGN_DT")))

# Create "keep" column with a value of 0 or 1 based on "start_month" and "end_month" comparison
longterm_merge = longterm_merge.withColumn("keep", 
                                       when((col("start_month") <= col("service_month")) & 
                                            (col("service_month") <= col("end_month")), lit(1))
                                       .otherwise(lit(0)))

# Create the "pre" indicator
longterm_merge = longterm_merge.withColumn("pre", 
                                       when((col("pre_start") <= col("service_month")) & 
                                            (col("service_month") <= col("pre_end")), 1)
                                       .otherwise(0))

# Create the "post" indicator
longterm_merge = longterm_merge.withColumn("post", 
                                       when((col("post_start") <= col("service_month")) & 
                                            (col("service_month") <= col("post_end")), 1)
                                       .otherwise(0))

# COMMAND ----------

# Select the desired columns
selected = longterm_merge.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

# Show the final result
selected.show(500)

# COMMAND ----------

print(longterm_merge.count())
filtered_df = longterm_merge.filter(longterm_merge["keep"] == 1)
print(filtered_df.count())

# Select the desired columns
selected = filtered_df.select("beneID", "state", "SRVC_BGN_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

selected.groupby('keep').count().orderBy(desc('count'),'keep').show()
selected.groupby('pre').count().orderBy(desc('count'),'pre').show()
selected.groupby('post').count().orderBy(desc('count'),'post').show()

#pre
#|  1|257452283|
#|  0|236373338|

# Show the final result
selected.show(500)

# COMMAND ----------

filtered_df.write.saveAsTable("dua_058828_spa240.paper1_stage2_longterm_12month", mode='overwrite') 

# COMMAND ----------

pharm2017 = spark.table("dua_058828_spa240.pharm_final2017")
pharm2018 = spark.table("dua_058828_spa240.pharm_final2018")
pharm2019 = spark.table("dua_058828_spa240.pharm_final2019")

# Union the three dataframes
pharm = pharm2017.union(pharm2018).union(pharm2019)

print(pharm.printSchema())

# COMMAND ----------

from pyspark.sql.functions import month, year, col, lit

#print(outpat.count())
pharm_merge = pharm.join(df, on=['beneID','state'], how='inner')
#print(outpat_merge.count())

# Map "SRVC_BGN_DT" to a "service_month" value from 1 to 36
pharm_merge = pharm_merge.withColumn("service_month", 
                                       (year(col("RX_FILL_DT")) - 2017) * 12 + month(col("RX_FILL_DT")))

# Create "keep" column with a value of 0 or 1 based on "start_month" and "end_month" comparison
pharm_merge = pharm_merge.withColumn("keep", 
                                       when((col("start_month") <= col("service_month")) & 
                                            (col("service_month") <= col("end_month")), lit(1))
                                       .otherwise(lit(0)))

# Create the "pre" indicator
pharm_merge = pharm_merge.withColumn("pre", 
                                       when((col("pre_start") <= col("service_month")) & 
                                            (col("service_month") <= col("pre_end")), 1)
                                       .otherwise(0))

# Create the "post" indicator
pharm_merge = pharm_merge.withColumn("post", 
                                       when((col("post_start") <= col("service_month")) & 
                                            (col("service_month") <= col("post_end")), 1)
                                       .otherwise(0))

# COMMAND ----------

# Select the desired columns
selected = pharm_merge.select("beneID", "state", "RX_FILL_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

# Show the final result
selected.show(500)

# COMMAND ----------

print(pharm_merge.count())
filtered_df = pharm_merge.filter(pharm_merge["keep"] == 1)
print(filtered_df.count())

# Select the desired columns
selected = filtered_df.select("beneID", "state", "RX_FILL_DT", "enrollMonth", "enrollYear", "start_month", "end_month", "pre_start", "pre_end", "post_start", "post_end", "service_month", "keep", "pre", "post")

selected.groupby('keep').count().orderBy(desc('count'),'keep').show()
selected.groupby('pre').count().orderBy(desc('count'),'pre').show()
selected.groupby('post').count().orderBy(desc('count'),'post').show()

# Show the final result
selected.show(500)

# COMMAND ----------

filtered_df.write.saveAsTable("dua_058828_spa240.paper1_stage2_pharm_12month", mode='overwrite') 