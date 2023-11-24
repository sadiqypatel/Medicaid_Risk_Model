# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
from pyspark.sql import SparkSession
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

# Define a list of month names
# header
month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr19_other_services_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "REV_CNTR_CD", "LINE_PRCDR_CD", "LINE_PRCDR_MDFR_CD_1", "SRVC_PRVDR_NPI", "SRVC_PRVDR_SPCLTY_CD", "LINE_BILLED_AMT", "LINE_MDCD_PD_AMT","LINE_MDCR_PD_AMT","LINE_COPAY_AMT")
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])
  exec(f"dfLine_{month_name} = df")


# COMMAND ----------

# Print the first 10 rows of each DataFrame
#for month_name in month_names:
#  exec(f"dfLine_{month_name}.show(10)")

# COMMAND ----------

#header
# Define a list of month names
month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr19_other_services_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "BILL_TYPE_CD", "SRVC_BGN_DT", "SRVC_END_DT", "DGNS_CD_1", "DGNS_CD_2", "POS_CD", "BLG_PRVDR_NPI", "BLG_PRVDR_SPCLTY_CD")
  df = df.where((col("CLM_TYPE_CD") == '1') | (col("CLM_TYPE_CD") == '3'))
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])  
  # Assign the DataFrame to a separate variable
  exec(f"dfHeader_{month_name} = df")
  

# COMMAND ----------

#dfHeader_01.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_02.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_03.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_04.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_05.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_06.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_07.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_08.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_09.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_10.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_11.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()
#dfHeader_12.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()

# COMMAND ----------

# Print the first 10 rows of each DataFrame
#for month_name in month_names:
#  exec(f"dfHeader_{month_name}.show(10)")

# COMMAND ----------

month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  header = f"dfHeader_{month_name}"
  line = f"dfLine_{month_name}"
# Perform inner join on three keys
  joined_df = eval(f"dfLine_{month_name}").join(eval(f"dfHeader_{month_name}"), 
                                 on=["BENE_ID", "STATE_CD", "CLM_ID"], 
                                 how="inner")
  exec(f"otherServices_{month_name} = joined_df")


# COMMAND ----------

#print(otherServices_01.count())
#print(dfHeader_01.count())
#print(dfLine_01.count())

#print(otherServices_02.count())
#print(dfHeader_02.count())
#print(dfLine_02.count())

#print(otherServices_03.count())
#print(dfHeader_03.count())
#print(dfLine_03.count())

#print(otherServices_01.count())
#print(otherServices_02.count())
#print(otherServices_03.count())
#print(otherServices_04.count())
#print(otherServices_05.count())
#print(otherServices_06.count())
#print(otherServices_07.count())
#print(otherServices_08.count())
#print(otherServices_09.count())
#print(otherServices_10.count())
#print(otherServices_11.count())
#print(otherServices_12.count())

# COMMAND ----------

otherServices2019 = otherServices_01.union(otherServices_02).union(otherServices_03).union(otherServices_04).union(otherServices_05).union(otherServices_06).union(otherServices_07).union(otherServices_08).union(otherServices_09).union(otherServices_10).union(otherServices_11).union(otherServices_12)

print((otherServices2019.count(), len(otherServices2019.columns)))
otherServices2019.write.saveAsTable("dua_058828_spa240.otherServices2019", mode='overwrite')

# COMMAND ----------

otherServices2019 = otherServices2019.withColumn("procNotNull", when(col("LINE_PRCDR_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("diagNotNull", when(col("DGNS_CD_1").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("npiNotNull", when(col("SRVC_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("specCodeNotNull", when(col("SRVC_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiSpecNotNull", when(col("BLG_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("lineBilledNotNull", when(col("LINE_BILLED_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineMdcdPaidNotNull", when(col("LINE_MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineMdcrPaidNotNull", when(col("LINE_MDCR_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineBeneCopayNotNull", when(col("LINE_COPAY_AMT").isNotNull(), lit(1)).otherwise(lit(0)))

otherServices2019 = otherServices2019.withColumn("LinePaidNotNull", when((col("LineBeneCopayNotNull")==1) | (col("LineMdcrPaidNotNull")==1) | (col("LineMdcdPaidNotNull")==1), lit(1)).otherwise(lit(0)))

# COMMAND ----------

otherServices2019.registerTempTable("missingTable")
missingSpend = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(LineBeneCopayNotNull) as LineBeneCopayNotNull, max(LineMdcrPaidNotNull) as LineMdcrPaidNotNull, max(LineMdcdPaidNotNull) as LineMdcdPaidNotNull, 
max(LinePaidNotNull) as LinePaidNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

missingSpend = missingSpend.withColumn("totalClaims", lit(1))

#aggregate to total 
missingSpend.registerTempTable("missingTotal")
missingSpendTotal = spark.sql('''
select distinct sum(LineBeneCopayNotNull) as LineBeneCopayNotNull, sum(LineMdcrPaidNotNull) as LineMdcrPaidNotNull, sum(LineMdcdPaidNotNull) as LineMdcdPaidNotNull, 
sum(LinePaidNotNull) as LinePaidNotNull, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingSpendTotal.show()

# COMMAND ----------

#aggregate to claim level to check missingness
otherServices2019.registerTempTable("missingTable")
missingTable = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(procNotNull) as procNotNull, max(diagNotNull) as diagNotNull, max(npiNotNull) as npiNotNull, max(specCodeNotNull) as specCodeNotNull, max(billNpiNotNull) as billNpiNotNull, max(billNpiSpecNotNull) as billNpiSpecNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

#measure how many are missing
missingTable = missingTable.withColumn("noMissing", when((col("procNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0)))
missingTable = missingTable.withColumn("totalClaims", lit(1))
#missingTable.groupBy('noMissing').count().show()
#missingTable.groupBy('totalClaims').count().show()

#aggregate to state
missingTable.registerTempTable("missingState")
missingTable = spark.sql('''
select distinct STATE_CD, sum(noMissing) as noMissing, sum(totalClaims) as totalClaims
FROM missingState 
GROUP BY STATE_CD; 
''')

#aggregate to total 
missingTable.registerTempTable("missingTotal")
missingTotal = spark.sql('''
select sum(noMissing) as noMissing, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingTable.write.saveAsTable("dua_058828_spa240.otherServicesMissing2019", mode='overwrite')

# COMMAND ----------

# Write DataFrame to CSV
#df = spark.table("dua_058828_spa240.otherServicesMissing2019")
#dbutils.fs.ls("s3://apcws301-transfer/dua/dua_058828/toSAS/")
#df.write.format('csv').option('header', 'true').mode('overwrite').save("dbfs:/mnt/dua/dua_058828/SPA240/files/ofState2019d.csv")
#dbutils.fs.ls("dbfs:/mnt/dua/dua_058828/SPA240/files")
#dbutils.fs.mv(f'dbfs:/mnt/dua/dua_058828/SPA240/files/ofState2019d.csv/', f's3://apcws301-transfer/dua/dua_058828/toSAS/ofState2019d.csv/', True)