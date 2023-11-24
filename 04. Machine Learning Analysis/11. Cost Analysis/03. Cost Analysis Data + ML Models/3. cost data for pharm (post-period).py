# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql.functions import month, year, col, lit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
from pyspark.sql import SparkSession
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# line item
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr17_rx_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "LINE_MDCD_PD_AMT", "LINE_MDCD_FFS_EQUIV_AMT")
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])
  exec(f"dfLine_{month_name} = df")
  
#header
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr17_rx_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "RX_FILL_DT", "MDCD_PD_AMT")
  df = df.where((col("CLM_TYPE_CD") == '1') | (col("CLM_TYPE_CD") == '3'))
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])  
  # Assign the DataFrame to a separate variable
  exec(f"dfHeader_{month_name} = df")
  
# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  header = f"dfHeader_{month_name}"
  line = f"dfLine_{month_name}"
# Perform inner join on three keys
  joined_df = eval(f"dfLine_{month_name}").join(eval(f"dfHeader_{month_name}"), 
                                 on=["BENE_ID", "STATE_CD", "CLM_ID"], 
                                 how="inner")
  exec(f"otherServices_{month_name} = joined_df")
  
pharm2017 = otherServices_01.union(otherServices_02).union(otherServices_03).union(otherServices_04).union(otherServices_05).union(otherServices_06).union(otherServices_07).union(otherServices_08).union(otherServices_09).union(otherServices_10).union(otherServices_11).union(otherServices_12)

# print((pharm2017.count(), len(pharm2017.columns)))

# COMMAND ----------

month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# line item
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr18_rx_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "LINE_MDCD_PD_AMT", "LINE_MDCD_FFS_EQUIV_AMT")
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])
  exec(f"dfLine_{month_name} = df")
  
#header
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr18_rx_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "RX_FILL_DT", "MDCD_PD_AMT")
  df = df.where((col("CLM_TYPE_CD") == '1') | (col("CLM_TYPE_CD") == '3'))
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])  
  # Assign the DataFrame to a separate variable
  exec(f"dfHeader_{month_name} = df")
  
# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  header = f"dfHeader_{month_name}"
  line = f"dfLine_{month_name}"
# Perform inner join on three keys
  joined_df = eval(f"dfLine_{month_name}").join(eval(f"dfHeader_{month_name}"), 
                                 on=["BENE_ID", "STATE_CD", "CLM_ID"], 
                                 how="inner")
  exec(f"otherServices_{month_name} = joined_df")
  
pharm2018 = otherServices_01.union(otherServices_02).union(otherServices_03).union(otherServices_04).union(otherServices_05).union(otherServices_06).union(otherServices_07).union(otherServices_08).union(otherServices_09).union(otherServices_10).union(otherServices_11).union(otherServices_12)

# print((pharm2018.count(), len(pharm2018.columns)))

# COMMAND ----------

month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# line item
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr19_rx_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "LINE_MDCD_PD_AMT", "LINE_MDCD_FFS_EQUIV_AMT")
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])
  exec(f"dfLine_{month_name} = df")
  
#header
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr19_rx_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "RX_FILL_DT", "MDCD_PD_AMT")
  df = df.where((col("CLM_TYPE_CD") == '1') | (col("CLM_TYPE_CD") == '3'))
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])  
  # Assign the DataFrame to a separate variable
  exec(f"dfHeader_{month_name} = df")
  
# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  header = f"dfHeader_{month_name}"
  line = f"dfLine_{month_name}"
# Perform inner join on three keys
  joined_df = eval(f"dfLine_{month_name}").join(eval(f"dfHeader_{month_name}"), 
                                 on=["BENE_ID", "STATE_CD", "CLM_ID"], 
                                 how="inner")
  exec(f"otherServices_{month_name} = joined_df")
  
pharm2019 = otherServices_01.union(otherServices_02).union(otherServices_03).union(otherServices_04).union(otherServices_05).union(otherServices_06).union(otherServices_07).union(otherServices_08).union(otherServices_09).union(otherServices_10).union(otherServices_11).union(otherServices_12)

# print((pharm2019.count(), len(pharm2019.columns)))

# COMMAND ----------

pharm_paper1_cost = pharm2019.union(pharm2018).union(pharm2017)
pharm_paper1_cost = pharm_paper1_cost.withColumnRenamed("BENE_ID", "beneID") \
       .withColumnRenamed("STATE_CD", "state")

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_sample")
df = df.select("beneID", "state", "enrollYear","enrollMonth")
# print((df.count(), len(df.columns)))
# print(df.printSchema())

# Assign values 1-24 based on enrollMonth and enrollYear

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
# df.show(500)

# COMMAND ----------

pharm_merge = pharm_paper1_cost.join(df, on=['beneID','state'], how='inner')

# Map "SRVC_BGN_DT" to a "service_month" value from 1 to 36
pharm_merge = pharm_merge.withColumn("service_month", 
                                       (year(col("RX_FILL_DT")) - 2017) * 12 + month(col("RX_FILL_DT")))

# Create the "post" indicator
pharm_merge = pharm_merge.withColumn("post", 
                                       when((col("post_start") <= col("service_month")) & 
                                            (col("service_month") <= col("post_end")), 1)
                                       .otherwise(0))

# COMMAND ----------

filtered_df = pharm_merge.filter(pharm_merge["post"] == 1)

# Keep rows with specific states
states_to_keep = ["AL", "ME", "MT", "VT", "WY", "IL"]
filtered_df = filtered_df.filter(col("state").isin(states_to_keep))
print(filtered_df.count())

filtered_df.write.saveAsTable("dua_058828_spa240.paper1_stage2_pharm_cost_15M", mode='overwrite') 

# COMMAND ----------

# print(pharm_merge.count())
# print(filtered_df.count())

# COMMAND ----------

# # Select the desired columns
# selected = pharm_merge.select("beneID", "state", "RX_FILL_DT", "enrollMonth", "enrollYear", "post_start", "post_end", "service_month", "post")

# # Show the final result
# selected.show(500)

# COMMAND ----------

# # Select the desired columns
# selected = filtered_df.select("beneID", "state", "RX_FILL_DT", "enrollMonth", "enrollYear", "post_start", "post_end", "service_month", "post")
# selected.groupby('post').count().orderBy(desc('count'),'post').show()

# # Show the final result
# selected.show(500)

# print(pharm_merge.count())
# print(selected.count())