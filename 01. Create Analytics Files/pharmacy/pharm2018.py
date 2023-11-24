# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
from pyspark.sql import SparkSession
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

# LINE
month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr18_rx_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "NDC", "DAYS_SUPPLY", "NEW_RX_REFILL_NUM", "BRND_GNRC_CD", "RSN_SRVC_CD", "LINE_BILLED_AMT", "LINE_MDCD_PD_AMT", "LINE_MDCR_PD_AMT", "LINE_COPAY_AMT")
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
  table_name = f"dua_058828.tafr18_rx_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "RX_FILL_DT", "PRSCRBNG_PRVDR_NPI", "BLG_PRVDR_NPI", "BLG_PRVDR_SPCLTY_CD", "MDCD_PD_AMT", "MDCD_COPAY_AMT", "BILLED_AMT")
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
  exec(f"pharm_{month_name} = joined_df")


# COMMAND ----------

print(pharm_01.count())
print(pharm_02.count())
print(pharm_03.count())
print(pharm_04.count())
print(pharm_05.count())
print(pharm_06.count())
print(pharm_07.count())
print(pharm_08.count())
print(pharm_09.count())
print(pharm_10.count())
print(pharm_11.count())
print(pharm_12.count())

# COMMAND ----------

pharm2018 = pharm_01.union(pharm_02).union(pharm_03).union(pharm_04).union(pharm_05).union(pharm_06).union(pharm_07).union(pharm_08).union(pharm_09).union(pharm_10).union(pharm_11).union(pharm_12)

print((pharm2018.count(), len(pharm2018.columns)))
pharm2018.write.saveAsTable("dua_058828_spa240.pharm2018", mode='overwrite')

# COMMAND ----------

pharm2018 = pharm2018.withColumn("rxDateNotNull", when(col("RX_FILL_DT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("ndcNotNull", when(col("NDC").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("daysSuppNotNull", when(col("DAYS_SUPPLY").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("brandNotNull", when(col("BRND_GNRC_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiSpecNotNull", when(col("BLG_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("rxNpiNotNull", when(col("PRSCRBNG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("lineBilledNotNull", when(col("LINE_BILLED_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineMdcdPaidNotNull", when(col("LINE_MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineMdcrPaidNotNull", when(col("LINE_MDCR_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("LineBeneCopayNotNull", when(col("LINE_COPAY_AMT").isNotNull(), lit(1)).otherwise(lit(0)))

pharm2018 = pharm2018.withColumn("LinePaidNotNull", when((col("LineBeneCopayNotNull")==1) | (col("LineMdcrPaidNotNull")==1) | (col("LineMdcdPaidNotNull")==1), lit(1)).otherwise(lit(0)))

# COMMAND ----------

pharm2018.registerTempTable("missingTable")
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

#create indicator for notNull

#aggregate to claim level to check missingness
pharm2018.registerTempTable("missingTable")
missingTable = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(rxDateNotNull) as rxDateNotNull, max(ndcNotNull) as ndcNotNull, max(daysSuppNotNull) as daysSuppNotNull, max(brandNotNull) as brandNotNull, max(billNpiNotNull) as billNpiNotNull, max(billNpiSpecNotNull) as billNpiSpecNotNull, max(rxNpiNotNull) as rxNpiNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

#measure how many are missing
missingTable = missingTable.withColumn("noMissing", when((col("rxDateNotNull")==1) & (col("ndcNotNull")==1) & (col("daysSuppNotNull")==1) & (col("brandNotNull")==1) & ((col("rxNpiNotNull")==1) | (col("billNpiSpecNotNull")==1) | (col("billNpiNotNull")==1)), lit(1)).otherwise(lit(0))).withColumn("noEssentialMissing", when((col("rxDateNotNull")==1) & (col("ndcNotNull")==1) & (col("daysSuppNotNull")==1) & ((col("rxNpiNotNull")==1) | (col("billNpiSpecNotNull")==1) | (col("billNpiNotNull")==1)), lit(1)).otherwise(lit(0))).withColumn("noMinMissing", when((col("rxDateNotNull")==1) & (col("ndcNotNull")==1) & (col("daysSuppNotNull")==1), lit(1)).otherwise(lit(0)))
missingTable = missingTable.withColumn("totalClaims", lit(1))
#missingTable.groupBy('noMissing').count().show()
#missingTable.groupBy('totalClaims').count().show()

#aggregate to state
missingTable.registerTempTable("missingState")
missingTable = spark.sql('''
select distinct STATE_CD, sum(noMissing) as noMissing, sum(noEssentialMissing) as noEssentialMissing, sum(noMinMissing) as noMinMissing, sum(totalClaims) as totalClaims    
FROM missingState 
GROUP BY STATE_CD; 
''')

#aggregate to total 
missingTable.registerTempTable("missingTotal")
missingTotal = spark.sql('''
select sum(noMissing) as noMissing, sum(noEssentialMissing) as noEssentialMissing, sum(noMinMissing) as noMinMissing, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingTable.write.saveAsTable("dua_058828_spa240.pharmMissing2018", mode='overwrite')

# COMMAND ----------

# Write DataFrame to CSV
#df = spark.table("dua_058828_spa240.pharmMissing2018")
#dbutils.fs.ls("s3://apcws301-transfer/dua/dua_058828/toSAS/")
#df.write.format('csv').option('header', 'true').mode('overwrite').save("dbfs:/mnt/dua/dua_058828/SPA240/files/pharm2018.csv")
#dbutils.fs.ls("dbfs:/mnt/dua/dua_058828/SPA240/files")
#dbutils.fs.mv(f'dbfs:/mnt/dua/dua_058828/SPA240/files/pharm2018.csv/', f's3://apcws301-transfer/dua/dua_058828/toSAS/pharm2018.csv/', True)