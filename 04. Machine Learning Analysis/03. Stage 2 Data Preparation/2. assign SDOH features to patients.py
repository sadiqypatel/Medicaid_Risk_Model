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

df = spark.table("dua_058828_spa240.demographic_stage2")
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

df = df.select("beneID", "state", "county")
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

county_missing = df.filter(df.county.isNull()).count() 

# Get the total number of rows in the DataFrame
total_rows = df.count()

# Calculate the proportion of missing values
missing_proportion = county_missing / total_rows

print(county_missing)
print(missing_proportion)

# COMMAND ----------

import pandas as pd
pandas_df = pd.read_sas("/dbfs/mnt/dua/dua_058828/SPA240/files/ahrq_sdoh_county_2020.sas7bdat")
# Convert byte literals to strings
pandas_df['COUNTYFIPS'] = pandas_df['COUNTYFIPS'].str.decode('utf-8')
pandas_df['STATE'] = pandas_df['STATE'].str.decode('utf-8')
pandas_df['STATEFIPS'] = pandas_df['STATEFIPS'].str.decode('utf-8')
print(pandas_df)

# COMMAND ----------

sdoh = spark.createDataFrame(pandas_df)
print((sdoh.count(), len(sdoh.columns)))

# COMMAND ----------

print((sdoh.count(), len(sdoh.columns)))
print(sdoh.printSchema())

# COMMAND ----------

sdoh = sdoh.select('STATE', 'STATEFIPS','COUNTYFIPS','AMFAR_MEDSAFAC_RATE', 'AMFAR_MEDAMATFAC_RATE', 'AMFAR_MEDMHFAC_RATE', 'CEN_POPDENSITY_COUNTY', 'ACS_PCT_INC50', 'ACS_PCT_HH_PUB_ASSIST', 'ACS_PCT_LT_HS', 'AHRF_PCT_GOOD_AQ', 'CDCW_INJURY_DTH_RATE', 'CDCW_ASSAULT_DTH_RATE', 'NEPHTN_ARSENIC_MEAN_POP', 'HIFLD_UC_RATE', 'EPAA_MEAN_WTD_PM25', 'CDCW_OPIOID_DTH_RATE', 'CDCW_DRUG_DTH_RATE', 'NEPHTN_HEATIND_100', 'AHRF_NURSE_PRACT_RATE', 'AHRF_PHYSICIAN_ASSIST_RATE', 'AHRF_CLIN_NURSE_SPEC_RATE', 'AHRF_ADV_NURSES_RATE')

print(sdoh.printSchema())

# COMMAND ----------

sdoh = sdoh.withColumn("aprnRate", sdoh.AHRF_NURSE_PRACT_RATE + sdoh.AHRF_PHYSICIAN_ASSIST_RATE + sdoh.AHRF_CLIN_NURSE_SPEC_RATE + sdoh.AHRF_ADV_NURSES_RATE)
sdoh = sdoh.drop("AHRF_NURSE_PRACT_RATE","AHRF_PHYSICIAN_ASSIST_RATE","AHRF_CLIN_NURSE_SPEC_RATE","AHRF_ADV_NURSES_RATE") 
print(sdoh.printSchema())

# COMMAND ----------

sdoh.show(10)

# COMMAND ----------

sdoh = sdoh.withColumnRenamed("STATE", "state").withColumnRenamed("STATEFIPS", "statefips").withColumnRenamed("COUNTYFIPS", "county").withColumnRenamed("AMFAR_MEDSAFAC_RATE", 'saServRate').withColumnRenamed("AMFAR_MEDAMATFAC_RATE", 'saFacRate').withColumnRenamed("AMFAR_MEDMHFAC_RATE", 'mhTreatRate').withColumnRenamed("CEN_POPDENSITY_COUNTY", 'popDensity').withColumnRenamed("ACS_PCT_INC50", 'povRate').withColumnRenamed("ACS_PCT_HH_PUB_ASSIST", 'publicAssistRate').withColumnRenamed("ACS_PCT_LT_HS", 'highSchoolGradRate').withColumnRenamed("AHRF_PCT_GOOD_AQ", 'goodAirDays').withColumnRenamed("CDCW_INJURY_DTH_RATE", 'injDeathRate').withColumnRenamed("CDCW_ASSAULT_DTH_RATE", 'assaultDeathRate').withColumnRenamed("NEPHTN_ARSENIC_MEAN_POP", 'waterQuality').withColumnRenamed("HIFLD_UC_RATE", 'urgentCareRate').withColumnRenamed("EPAA_MEAN_WTD_PM25", 'airPm25Rate').withColumnRenamed("CDCW_OPIOID_DTH_RATE", 'opioidDeathRate').withColumnRenamed("CDCW_DRUG_DTH_RATE", 'drugdeathRate').withColumnRenamed("NEPHTN_HEATIND_100", '100HeatDays')
print(sdoh.printSchema())

# COMMAND ----------

columns_to_check = ["state", "saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "assaultDeathRate", "waterQuality", "urgentCareRate", "airPm25Rate", "opioidDeathRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Get the total number of rows in the DataFrame
total_rows = sdoh.count()

# Calculate the proportion of missing values in each column
for col_name in columns_to_check:
    missing_count = sdoh.filter(F.col(col_name).isNull()).count()
    missing_proportion = missing_count / total_rows * 100
    print(f"Percentage of missing values in {col_name}: {missing_proportion}")

for col_name in columns_to_check:
    print(f"Number of null values in {col_name}: {sdoh.filter(F.col(col_name).isNull()).count()}")

# COMMAND ----------

sdoh = sdoh.drop("assaultDeathRate","waterQuality","opioidDeathRate", "airPm25Rate")
print(sdoh.printSchema())

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define a Window partitioned by 'State'
window = Window.partitionBy('state')

columns_to_impute = ["saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "urgentCareRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Calculate the mean for each column over the Window, and replace nulls with the mean
for col_name in columns_to_impute:
    mean_col = F.mean(sdoh[col_name]).over(window)
    sdoh = sdoh.withColumn(col_name, F.when(F.col(col_name).isNull(), mean_col).otherwise(F.col(col_name)))

    # Calculate the overall column mean
    overall_mean = sdoh.select(F.mean(F.col(col_name)).alias('overall_mean')).collect()[0]['overall_mean']

    # Replace any remaining nulls with the overall column mean
    sdoh = sdoh.withColumn(col_name, F.when(F.col(col_name).isNull(), overall_mean).otherwise(F.col(col_name)))
    
print(sdoh.printSchema()) 

# COMMAND ----------

sdoh = sdoh.drop("state","statefips")
print(sdoh.printSchema()) 

# COMMAND ----------

# Calculate the proportion of missing values in each column

columns_to_check = ["saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "urgentCareRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Calculate the mean for each column over the Window, and replace nulls with the mean
for col_name in columns_to_check:
    print(f"Number of null values in {col_name}: {sdoh.filter(F.col(col_name).isNull()).count()}")

# COMMAND ----------

print((sdoh.count(), len(sdoh.columns)))

# Count the number of unique values in the 'county' column
unique_count = sdoh.select("county").distinct().count()
print("Number of unique counties:", unique_count)

# COMMAND ----------

merged_df = df.join(sdoh, on="county", how="left")

# Calculate the proportion of missing values in each column

columns_to_check = ["saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "urgentCareRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Get the total number of rows in the DataFrame
total_rows = merged_df.count()

# Calculate the proportion of missing values in each column
for col_name in columns_to_check:
    missing_count = merged_df.filter(F.col(col_name).isNull()).count()
    missing_proportion = missing_count / total_rows * 100
    print(f"Percentage of missing values in {col_name}: {missing_proportion}")

# COMMAND ----------

# Define a Window partitioned by 'State'
window = Window.partitionBy('state')

# List of columns to impute
columns_to_impute = ["saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "urgentCareRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Calculate the mean for each column over the Window, and replace nulls with the mean
for col_name in columns_to_impute:
    mean_col = F.mean(merged_df[col_name]).over(window)
    merged_df = merged_df.withColumn(col_name, F.when(F.col(col_name).isNull(), mean_col).otherwise(F.col(col_name)))

    # Calculate the overall column mean
    overall_mean = merged_df.select(F.mean(F.col(col_name)).alias('overall_mean')).collect()[0]['overall_mean']

    # Replace any remaining nulls with the overall column mean
    merged_df = merged_df.withColumn(col_name, F.when(F.col(col_name).isNull(), overall_mean).otherwise(F.col(col_name)))

# COMMAND ----------

columns_to_check = ["saServRate", "saFacRate", "mhTreatRate", "popDensity", "povRate", "publicAssistRate", "highSchoolGradRate", "goodAirDays", "injDeathRate", "urgentCareRate", "drugdeathRate", "100HeatDays", "aprnRate"]

# Calculate the mean for each column over the Window, and replace nulls with the mean
for col_name in columns_to_check:
    print(f"Number of null values in {col_name}: {merged_df.filter(F.col(col_name).isNull()).count()}")

# COMMAND ----------

merged_df.write.saveAsTable("dua_058828_spa240.sdoh_features_stage2", mode="overwrite")

# COMMAND ----------

df = spark.table("dua_058828_spa240.sdoh_features_stage2")
print(df.count())