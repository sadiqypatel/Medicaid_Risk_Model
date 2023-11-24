# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
from pyspark.sql import SparkSession
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

# Define a list of month names
# line
month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# Loop through the list of month names and create separate DataFrame variables
for month_name in month_names:
  # Get the DataFrame for the current month
  table_name = f"dua_058828.tafr18_long_term_line_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "LINE_NUM", "REV_CNTR_CD", "SRVC_PRVDR_NPI", "SRVC_PRVDR_SPCLTY_CD", "LINE_MDCD_PD_AMT", "LINE_MDCD_FFS_EQUIV_AMT")
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
  table_name = f"dua_058828.tafr18_long_term_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "BILL_TYPE_CD", "SRVC_BGN_DT", "SRVC_END_DT", "DGNS_CD_1", "DGNS_CD_2", "DGNS_CD_3", "BLG_PRVDR_NPI", "BLG_PRVDR_SPCLTY_CD", "BILLED_AMT", "MDCR_PD_AMT", "MDCD_PD_AMT", "COPAY_AMT")  
  df = df.where((col("CLM_TYPE_CD") == '1') | (col("CLM_TYPE_CD") == '3'))
  df = df.dropna(subset=["BENE_ID"])
  df = df.dropna(subset=["STATE_CD"])  
  # Assign the DataFrame to a separate variable
  exec(f"dfHeader_{month_name} = df")
  

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
  exec(f"longTerm_{month_name} = joined_df")


# COMMAND ----------

#print(longTerm_01.count())
#print(dfHeader_01.count())
#print(dfLine_01.count())

#print(longTerm_02.count())
#print(dfHeader_02.count())
#print(dfLine_02.count())

#print(longTerm_03.count())
#print(dfHeader_03.count())
#print(dfLine_03.count())

print("START HERE")
print(longTerm_01.count())
print(longTerm_02.count())
print(longTerm_03.count())
print(longTerm_04.count())
print(longTerm_05.count())
print(longTerm_06.count())
print(longTerm_07.count())
print(longTerm_08.count())
print(longTerm_09.count())
print(longTerm_10.count())
print(longTerm_11.count())
print(longTerm_12.count())

# COMMAND ----------

longTerm2018 = longTerm_01.union(longTerm_02).union(longTerm_03).union(longTerm_04).union(longTerm_05).union(longTerm_06).union(longTerm_07).union(longTerm_08).union(longTerm_09).union(longTerm_10).union(longTerm_11).union(longTerm_12)

print((longTerm2018.count(), len(longTerm2018.columns)))
longTerm2018.write.saveAsTable("dua_058828_spa240.longTerm2018", mode='overwrite')

# COMMAND ----------

longTerm2018.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()

# COMMAND ----------

#create indicator for notNull
longTerm2018 = longTerm2018.withColumn("diagNotNull", when(col("DGNS_CD_1").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("npiNotNull", when(col("SRVC_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("specCodeNotNull", when(col("SRVC_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("revNotNull", when(col("REV_CNTR_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("startDateNotNull", when(col("SRVC_BGN_DT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("endDateNotNull", when(col("SRVC_END_DT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiSpecNotNull", when(col("BLG_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("linePaidNotNull", when(col("LINE_MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headBilledNotNull", when(col("BILLED_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headMdcrNotNull", when(col("MDCR_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headMdcdNotNull", when(col("MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headBeneNotNull", when(col("COPAY_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("lineFfsEquivAmountNN", when(col("LINE_MDCD_FFS_EQUIV_AMT").isNotNull(), lit(1)).otherwise(lit(0)))

longTerm2018 = longTerm2018.withColumn("headTotPaidNotNull", when((col("headMdcrNotNull")==1) | (col("headMdcdNotNull")==1) | (col("headBeneNotNull")==1), lit(1)).otherwise(lit(0)))

# COMMAND ----------

longTerm2018.registerTempTable("missingTable")
missingSpend = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(linePaidNotNull) as linePaidNotNull, max(headMdcrNotNull) as headMdcrNotNull, max(headMdcdNotNull) as headMdcdNotNull, 
max(headTotPaidNotNull) as headTotPaidNotNull, max(lineFfsEquivAmountNN) as lineFfsEquivAmountNN, sum(headBeneNotNull) as headBeneNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

missingSpend = missingSpend.withColumn("totalClaims", lit(1))

#aggregate to total 
missingSpend.registerTempTable("missingTotal")
missingSpendTotal = spark.sql('''
select sum(linePaidNotNull) as linePaidNotNull, sum(lineFfsEquivAmountNN) as lineFfsEquivAmountNN, sum(headMdcrNotNull) as headMdcrNotNull, sum(headMdcdNotNull) as headMdcdNotNull, sum(headBeneNotNull) as headBeneNotNull, sum(headTotPaidNotNull) as headTotPaidNotNull, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingSpendTotal.show()

# COMMAND ----------

#aggregate to claim level to check missingness
longTerm2018.registerTempTable("missingTable")
missingTable = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(revNotNull) as revNotNull, max(diagNotNull) as diagNotNull, max(npiNotNull) as npiNotNull, max(startDateNotNull) as startDateNotNull, max(endDateNotNull) as endDateNotNull, max(specCodeNotNull) as specCodeNotNull, max(billNpiNotNull) as billNpiNotNull, max(billNpiSpecNotNull) as billNpiSpecNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

#measure how many are missing
missingTable = missingTable.withColumn("noMisisng", when((col("startDateNotNull")==1) &  (col("endDateNotNull")==1) & (col("revNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0))).withColumn("allButRev", when((col("startDateNotNull")==1) &  (col("endDateNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0)))

missingTable = missingTable.withColumn("totalClaims", lit(1))
#missingTable.groupBy('noMisisng').count().show()
#missingTable.groupBy('totalClaims').count().show()

#aggregate to state
missingTable.registerTempTable("missingState")
missingTable = spark.sql('''
select distinct STATE_CD, sum(noMisisng) as noMisisng, sum(allButRev) as allButRev, sum(totalClaims) as totalClaims
FROM missingState 
GROUP BY STATE_CD; 
''')

#aggregate to total 
missingTable.registerTempTable("missingTotal")
missingTotal = spark.sql('''
select sum(noMisisng) as noMisisng, sum(allButRev) as allButRev, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingTable.write.saveAsTable("dua_058828_spa240.longTermMissing2018", mode='overwrite')

# COMMAND ----------

missingTotal.show()

# COMMAND ----------

# Write DataFrame to CSV
#df = spark.table("dua_058828_spa240.longTermMissing2018")
#dbutils.fs.ls("s3://apcws301-transfer/dua/dua_058828/toSAS/")
#df.write.format('csv').option('header', 'true').mode('overwrite').save("dbfs:/mnt/dua/dua_058828/SPA240/files/longterm2018.csv")
#dbutils.fs.ls("dbfs:/mnt/dua/dua_058828/SPA240/files")
#dbutils.fs.mv(f'dbfs:/mnt/dua/dua_058828/SPA240/files/longterm2018.csv/', f's3://apcws301-transfer/dua/dua_058828/toSAS/longterm2018.csv/', True)