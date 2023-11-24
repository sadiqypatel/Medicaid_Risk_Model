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
  table_name = f"dua_058828.tafr19_inpatient_line_{month_name}"
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
  table_name = f"dua_058828.tafr19_inpatient_header_{month_name}"
  df = spark.table(table_name).select("BENE_ID", "STATE_CD", "CLM_ID", "CLM_TYPE_CD", "BILL_TYPE_CD", "SRVC_BGN_DT", "SRVC_END_DT", "PRCDR_CD_1", "DGNS_CD_1", "DGNS_CD_2", "DGNS_CD_3", "DRG_CD", "BLG_PRVDR_NPI", "BLG_PRVDR_SPCLTY_CD", "BILLED_AMT", "MDCR_PD_AMT", "MDCD_PD_AMT", "MDCD_COPAY_AMT")
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
  exec(f"inpatient_{month_name} = joined_df")


# COMMAND ----------

#print(inpatient_01.count())
#print(dfHeader_01.count())
#print(dfLine_01.count())

#print(inpatient_02.count())
#print(dfHeader_02.count())
#print(dfLine_02.count())

#print(inpatient_03.count())
#print(dfHeader_03.count())
#print(dfLine_03.count())

print('START HERE')
print(inpatient_01.count())
print(inpatient_02.count())
print(inpatient_03.count())
print(inpatient_04.count())
print(inpatient_05.count())
print(inpatient_06.count())
print(inpatient_07.count())
print(inpatient_08.count())
print(inpatient_09.count())
print(inpatient_10.count())
print(inpatient_11.count())
print(inpatient_12.count())

# COMMAND ----------

inpatient2019 = inpatient_01.union(inpatient_02).union(inpatient_03).union(inpatient_04).union(inpatient_05).union(inpatient_06).union(inpatient_07).union(inpatient_08).union(inpatient_09).union(inpatient_10).union(inpatient_11).union(inpatient_12)

print((inpatient2019.count(), len(inpatient2019.columns)))
inpatient2019.write.saveAsTable("dua_058828_spa240.inpatient2019", mode='overwrite')

# COMMAND ----------

inpatient2019.groupBy('CLM_TYPE_CD').count().orderBy(desc('count'),'CLM_TYPE_CD').show()

# COMMAND ----------

#create indicator for notNull
inpatient2019 = inpatient2019.withColumn("procNotNull", when(col("PRCDR_CD_1").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("diagNotNull", when(col("DGNS_CD_1").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("npiNotNull", when(col("SRVC_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("specCodeNotNull", when(col("SRVC_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("drgNotNull", when(col("DRG_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("startDateNotNull", when(col("SRVC_BGN_DT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("endDateNotNull", when(col("SRVC_END_DT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiNotNull", when(col("BLG_PRVDR_NPI").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("billNpiSpecNotNull", when(col("BLG_PRVDR_SPCLTY_CD").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("linePaidNotNull", when(col("LINE_MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headBilledNotNull", when(col("BILLED_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headMdcrNotNull", when(col("MDCR_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headMdcdNotNull", when(col("MDCD_PD_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("headBeneNotNull", when(col("MDCD_COPAY_AMT").isNotNull(), lit(1)).otherwise(lit(0))).withColumn("lineFfsEquivAmountNN", when(col("LINE_MDCD_FFS_EQUIV_AMT").isNotNull(), lit(1)).otherwise(lit(0)))

inpatient2019 = inpatient2019.withColumn("headTotPaidNotNull", when((col("headMdcrNotNull")==1) | (col("headMdcdNotNull")==1) | (col("headBeneNotNull")==1), lit(1)).otherwise(lit(0)))

# COMMAND ----------

inpatient2019.groupBy('lineFfsEquivAmountNN').count().orderBy(desc('count'),'lineFfsEquivAmountNN').show()
inpatient2019.groupBy('headTotPaidNotNull').count().orderBy(desc('count'),'headTotPaidNotNull').show()

inpatient2019.registerTempTable("missingTable")
missingSpend = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(linePaidNotNull) as linePaidNotNull, max(headMdcrNotNull) as headMdcrNotNull, max(headMdcdNotNull) as headMdcdNotNull, 
max(headTotPaidNotNull) as headTotPaidNotNull, max(lineFfsEquivAmountNN) as lineFfsEquivAmountNN
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

missingSpend = missingSpend.withColumn("totalClaims", lit(1))

#aggregate to total 
missingSpend.registerTempTable("missingTotal")
missingSpendTotal = spark.sql('''
select sum(linePaidNotNull) as linePaidNotNull, sum(lineFfsEquivAmountNN) as lineFfsEquivAmountNN, sum(headMdcrNotNull) as headMdcrNotNull, sum(headMdcdNotNull) as headMdcdNotNull, sum(headTotPaidNotNull) as headTotPaidNotNull, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingSpendTotal.show()

# COMMAND ----------

#aggregate to claim level to check missingness
inpatient2019.registerTempTable("missingTable")
missingTable = spark.sql('''
select distinct BENE_ID, STATE_CD, CLM_ID, max(procNotNull) as procNotNull, max(npiNotNull) as npiNotNull, max(specCodeNotNull) as specCodeNotNull, 
max(startDateNotNull) as startDateNotNull, max(endDateNotNull) as endDateNotNull, max(drgNotNull) as drgNotNull, max(diagNotNull) as diagNotNull, max(billNpiNotNull) as billNpiNotNull, max(billNpiSpecNotNull) as billNpiSpecNotNull
FROM missingTable 
GROUP BY BENE_ID, STATE_CD, CLM_ID; 
''')

#measure how many are missing
missingTable = missingTable.withColumn("all", when((col("procNotNull")==1) & (col("startDateNotNull")==1) & (col("endDateNotNull")==1) & (col("drgNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0))).withColumn("noProc", when((col("startDateNotNull")==1) & (col("endDateNotNull")==1) & (col("drgNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0))).withColumn("noProcDrg", when((col("startDateNotNull")==1) & (col("endDateNotNull")==1) & (col("diagNotNull")==1) & ((col("npiNotNull")==1) | (col("specCodeNotNull")==1) | (col("billNpiNotNull")==1) | (col("billNpiSpecNotNull")==1)), lit(1)).otherwise(lit(0)))

missingTable = missingTable.withColumn("totalClaims", lit(1))

#aggregate to state
missingTable.registerTempTable("missingState")
missingTable = spark.sql('''
select distinct STATE_CD, sum(all) as all, sum(noProc) as noProc, sum(noProcDrg) as noProcDrg, sum(totalClaims) as totalClaims
FROM missingState 
GROUP BY STATE_CD; 
''')

#aggregate to total 
missingTable.registerTempTable("missingTotal")
missingTotal = spark.sql('''
select sum(all) as all, sum(noProc) as noProc, sum(noProcDrg) as noProcDrg, sum(totalClaims) as totalClaims
FROM missingTotal ; 
''')

missingTable.write.saveAsTable("dua_058828_spa240.inpatientMissing2019", mode='overwrite')

# COMMAND ----------

missingTotal.show()

# COMMAND ----------

# Write DataFrame to CSV
#df = spark.table("dua_058828_spa240.inpatientMissing2019")
#dbutils.fs.ls("s3://apcws301-transfer/dua/dua_058828/toSAS/")
#df.write.format('csv').option('header', 'true').mode('overwrite').save("dbfs:/mnt/dua/dua_058828/SPA240/files/inpatient2019.csv")
#dbutils.fs.ls("dbfs:/mnt/dua/dua_058828/SPA240/files")
#dbutils.fs.mv(f'dbfs:/mnt/dua/dua_058828/SPA240/files/inpatient2019.csv/', f's3://apcws301-transfer/dua/dua_058828/toSAS/inpatient2019.csv/', True)