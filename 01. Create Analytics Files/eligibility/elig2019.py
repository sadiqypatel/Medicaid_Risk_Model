# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

df = spark.sql("select BENE_ID, STATE_CD, MDCD_ENRLMT_DAYS_01, MDCD_ENRLMT_DAYS_02, MDCD_ENRLMT_DAYS_03, MDCD_ENRLMT_DAYS_04, MDCD_ENRLMT_DAYS_05, MDCD_ENRLMT_DAYS_06, MDCD_ENRLMT_DAYS_07, MDCD_ENRLMT_DAYS_08, MDCD_ENRLMT_DAYS_09, MDCD_ENRLMT_DAYS_10, MDCD_ENRLMT_DAYS_11, MDCD_ENRLMT_DAYS_12, CHIP_ENRLMT_DAYS_01, CHIP_ENRLMT_DAYS_02, CHIP_ENRLMT_DAYS_03, CHIP_ENRLMT_DAYS_04, CHIP_ENRLMT_DAYS_05, CHIP_ENRLMT_DAYS_06, CHIP_ENRLMT_DAYS_07, CHIP_ENRLMT_DAYS_08, CHIP_ENRLMT_DAYS_09, CHIP_ENRLMT_DAYS_10, CHIP_ENRLMT_DAYS_11, CHIP_ENRLMT_DAYS_12, DUAL_ELGBL_CD_01, DUAL_ELGBL_CD_02, DUAL_ELGBL_CD_03, DUAL_ELGBL_CD_04, DUAL_ELGBL_CD_05, DUAL_ELGBL_CD_06, DUAL_ELGBL_CD_07, DUAL_ELGBL_CD_08, DUAL_ELGBL_CD_09, DUAL_ELGBL_CD_10, DUAL_ELGBL_CD_11, DUAL_ELGBL_CD_12, MC_PLAN_TYPE_CD_01, MC_PLAN_TYPE_CD_02, MC_PLAN_TYPE_CD_03, MC_PLAN_TYPE_CD_04, MC_PLAN_TYPE_CD_05, MC_PLAN_TYPE_CD_06, MC_PLAN_TYPE_CD_07, MC_PLAN_TYPE_CD_08, MC_PLAN_TYPE_CD_09, MC_PLAN_TYPE_CD_10, MC_PLAN_TYPE_CD_11, MC_PLAN_TYPE_CD_12, CHIP_CD_01, CHIP_CD_02, CHIP_CD_03, CHIP_CD_04, CHIP_CD_05, CHIP_CD_06, CHIP_CD_07, CHIP_CD_08, CHIP_CD_09, CHIP_CD_10, CHIP_CD_11, CHIP_CD_12 from dua_058828.tafr19_demog_elig_base")

# COMMAND ----------

df = df.dropna(subset=["BENE_ID"])
df = df.dropna(subset=["STATE_CD"])

# COMMAND ----------

print((df.count(), len(df.columns)))

# COMMAND ----------

df_medDays = df.select('BENE_ID','STATE_CD','MDCD_ENRLMT_DAYS_01','MDCD_ENRLMT_DAYS_02','MDCD_ENRLMT_DAYS_03','MDCD_ENRLMT_DAYS_04','MDCD_ENRLMT_DAYS_05','MDCD_ENRLMT_DAYS_06','MDCD_ENRLMT_DAYS_07','MDCD_ENRLMT_DAYS_08','MDCD_ENRLMT_DAYS_09','MDCD_ENRLMT_DAYS_10','MDCD_ENRLMT_DAYS_11','MDCD_ENRLMT_DAYS_12')

df_medDays = df_medDays.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", 'state').withColumnRenamed("MDCD_ENRLMT_DAYS_01", 'jan').withColumnRenamed("MDCD_ENRLMT_DAYS_02", 'feb').withColumnRenamed("MDCD_ENRLMT_DAYS_03", 'mar').withColumnRenamed("MDCD_ENRLMT_DAYS_04", 'apr').withColumnRenamed("MDCD_ENRLMT_DAYS_05", 'may').withColumnRenamed("MDCD_ENRLMT_DAYS_06", 'jun').withColumnRenamed("MDCD_ENRLMT_DAYS_07", 'jul').withColumnRenamed("MDCD_ENRLMT_DAYS_08", 'aug').withColumnRenamed("MDCD_ENRLMT_DAYS_09", 'sep').withColumnRenamed("MDCD_ENRLMT_DAYS_10", 'oct').withColumnRenamed("MDCD_ENRLMT_DAYS_11", 'nov').withColumnRenamed("MDCD_ENRLMT_DAYS_12", 'dec')


df_medDays.show(25)

# COMMAND ----------

df_medDays.registerTempTable("newtable")
df_medDays = spark.sql('''
select distinct beneID, state, sum(jan) as jan, sum(feb) as feb, sum(mar) as mar, sum(apr) as apr, sum(may) as may, sum(jun) as jun, sum(jul) as jul, sum(aug) as aug, sum(sep) as sep, sum(oct) as oct, sum(nov) as nov, sum(dec) as dec
FROM newtable 
GROUP BY beneID, state; 
''')

df_medDays.show(10)

# COMMAND ----------

df_medDays = df_medDays.withColumnRenamed("jan","1").withColumnRenamed("feb","2").withColumnRenamed("mar","3").withColumnRenamed("apr","4").withColumnRenamed("may","5").withColumnRenamed("jun","6").withColumnRenamed("jul","7").withColumnRenamed("aug","8").withColumnRenamed("sep","9").withColumnRenamed("oct","10").withColumnRenamed("nov","11").withColumnRenamed("dec","12")
#df_medDays.show()
df_medDays = df_medDays.to_koalas().melt(id_vars=['beneID','state'], value_vars=['1','2','3','4','5','6','7','8','9','10','11','12']).to_spark().withColumnRenamed("variable", 'month').withColumnRenamed("value", 'medicaidDays')
df_medDays.show(50)

# COMMAND ----------

print((df_medDays.count(), len(df_medDays.columns)))
df_medDays = df_medDays.dropDuplicates(["beneID","state","month"])
print((df_medDays.count(), len(df_medDays.columns)))

# COMMAND ----------

df_chipDays = df.select('BENE_ID','STATE_CD','CHIP_ENRLMT_DAYS_01','CHIP_ENRLMT_DAYS_02','CHIP_ENRLMT_DAYS_03','CHIP_ENRLMT_DAYS_04','CHIP_ENRLMT_DAYS_05','CHIP_ENRLMT_DAYS_06','CHIP_ENRLMT_DAYS_07','CHIP_ENRLMT_DAYS_08','CHIP_ENRLMT_DAYS_09','CHIP_ENRLMT_DAYS_10','CHIP_ENRLMT_DAYS_11','CHIP_ENRLMT_DAYS_12')

df_chipDays = df_chipDays.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", 'state').withColumnRenamed("CHIP_ENRLMT_DAYS_01", 'jan').withColumnRenamed("CHIP_ENRLMT_DAYS_02", 'feb').withColumnRenamed("CHIP_ENRLMT_DAYS_03", 'mar').withColumnRenamed("CHIP_ENRLMT_DAYS_04", 'apr').withColumnRenamed("CHIP_ENRLMT_DAYS_05", 'may').withColumnRenamed("CHIP_ENRLMT_DAYS_06", 'jun').withColumnRenamed("CHIP_ENRLMT_DAYS_07", 'jul').withColumnRenamed("CHIP_ENRLMT_DAYS_08", 'aug').withColumnRenamed("CHIP_ENRLMT_DAYS_09", 'sep').withColumnRenamed("CHIP_ENRLMT_DAYS_10", 'oct').withColumnRenamed("CHIP_ENRLMT_DAYS_11", 'nov').withColumnRenamed("CHIP_ENRLMT_DAYS_12", 'dec')

df_chipDays.show(25)

# COMMAND ----------

df_chipDays.registerTempTable("newtable")
df_chipDays = spark.sql('''
select distinct beneID, state, sum(jan) as jan, sum(feb) as feb, sum(mar) as mar, sum(apr) as apr, sum(may) as may, sum(jun) as jun, sum(jul) as jul, sum(aug) as aug, sum(sep) as sep, sum(oct) as oct, sum(nov) as nov, sum(dec) as dec
FROM newtable 
GROUP BY beneID, state; 
''')

df_chipDays.show()

# COMMAND ----------

df_chipDays = df_chipDays.withColumnRenamed("jan","1").withColumnRenamed("feb","2").withColumnRenamed("mar","3").withColumnRenamed("apr","4").withColumnRenamed("may","5").withColumnRenamed("jun","6").withColumnRenamed("jul","7").withColumnRenamed("aug","8").withColumnRenamed("sep","9").withColumnRenamed("oct","10").withColumnRenamed("nov","11").withColumnRenamed("dec","12")
#df_medDays.show()
df_chipDays = df_chipDays.to_koalas().melt(id_vars=['beneID','state'], value_vars=['1','2','3','4','5','6','7','8','9','10','11','12']).to_spark().withColumnRenamed("variable", 'month').withColumnRenamed("value", 'chipDays')
df_chipDays.show(10)

# COMMAND ----------

print((df_chipDays.count(), len(df_chipDays.columns)))
df_chipDays = df_chipDays.dropDuplicates(["beneID","state","month"])
print((df_chipDays.count(), len(df_chipDays.columns)))

# COMMAND ----------

df_dual = df.select('BENE_ID','STATE_CD','DUAL_ELGBL_CD_01','DUAL_ELGBL_CD_02','DUAL_ELGBL_CD_03','DUAL_ELGBL_CD_04','DUAL_ELGBL_CD_05','DUAL_ELGBL_CD_06','DUAL_ELGBL_CD_07','DUAL_ELGBL_CD_08','DUAL_ELGBL_CD_09','DUAL_ELGBL_CD_10','DUAL_ELGBL_CD_11','DUAL_ELGBL_CD_12')

df_dual = df_dual.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", 'state').withColumnRenamed("DUAL_ELGBL_CD_01", 'jan').withColumnRenamed("DUAL_ELGBL_CD_02", 'feb').withColumnRenamed("DUAL_ELGBL_CD_03", 'mar').withColumnRenamed("DUAL_ELGBL_CD_04", 'apr').withColumnRenamed("DUAL_ELGBL_CD_05", 'may').withColumnRenamed("DUAL_ELGBL_CD_06", 'jun').withColumnRenamed("DUAL_ELGBL_CD_07", 'jul').withColumnRenamed("DUAL_ELGBL_CD_08", 'aug').withColumnRenamed("DUAL_ELGBL_CD_09", 'sep').withColumnRenamed("DUAL_ELGBL_CD_10", 'oct').withColumnRenamed("DUAL_ELGBL_CD_11", 'nov').withColumnRenamed("DUAL_ELGBL_CD_12", 'dec')

df_dual.show(25)

# COMMAND ----------

df_dual.registerTempTable("newtable")
df_dual = spark.sql('''
select distinct beneID, state, max(jan) as jan, max(feb) as feb, max(mar) as mar, max(apr) as apr, max(may) as may, max(jun) as jun, max(jul) as jul, max(aug) as aug, max(sep) as sep, max(oct) as oct, max(nov) as nov, max(dec) as dec
FROM newtable 
GROUP BY beneID, state; 
''')

df_dual.show()

# COMMAND ----------

df_dual = df_dual.withColumnRenamed("jan","1").withColumnRenamed("feb","2").withColumnRenamed("mar","3").withColumnRenamed("apr","4").withColumnRenamed("may","5").withColumnRenamed("jun","6").withColumnRenamed("jul","7").withColumnRenamed("aug","8").withColumnRenamed("sep","9").withColumnRenamed("oct","10").withColumnRenamed("nov","11").withColumnRenamed("dec","12")
#df_medDays.show()
df_dual = df_dual.to_koalas().melt(id_vars=['beneID','state'], value_vars=['1','2','3','4','5','6','7','8','9','10','11','12']).to_spark().withColumnRenamed("variable", 'month').withColumnRenamed("value", 'dualStatus')
df_dual.show(50)

# COMMAND ----------

print((df_dual.count(), len(df_dual.columns)))
df_dual = df_dual.dropDuplicates(["beneID","state","month"])
print((df_dual.count(), len(df_dual.columns)))

# COMMAND ----------

df_mcPlanType = df.select('BENE_ID','STATE_CD','MC_PLAN_TYPE_CD_01','MC_PLAN_TYPE_CD_02','MC_PLAN_TYPE_CD_03','MC_PLAN_TYPE_CD_04','MC_PLAN_TYPE_CD_05','MC_PLAN_TYPE_CD_06','MC_PLAN_TYPE_CD_07','MC_PLAN_TYPE_CD_08','MC_PLAN_TYPE_CD_09','MC_PLAN_TYPE_CD_10','MC_PLAN_TYPE_CD_11','MC_PLAN_TYPE_CD_12')

df_mcPlanType = df_mcPlanType.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", 'state').withColumnRenamed("MC_PLAN_TYPE_CD_01", 'jan').withColumnRenamed("MC_PLAN_TYPE_CD_02", 'feb').withColumnRenamed("MC_PLAN_TYPE_CD_03", 'mar').withColumnRenamed("MC_PLAN_TYPE_CD_04", 'apr').withColumnRenamed("MC_PLAN_TYPE_CD_05", 'may').withColumnRenamed("MC_PLAN_TYPE_CD_06", 'jun').withColumnRenamed("MC_PLAN_TYPE_CD_07", 'jul').withColumnRenamed("MC_PLAN_TYPE_CD_08", 'aug').withColumnRenamed("MC_PLAN_TYPE_CD_09", 'sep').withColumnRenamed("MC_PLAN_TYPE_CD_10", 'oct').withColumnRenamed("MC_PLAN_TYPE_CD_11", 'nov').withColumnRenamed("MC_PLAN_TYPE_CD_12", 'dec')

df_mcPlanType.show(25)

# COMMAND ----------

df_mcPlanType.registerTempTable("newtable")
df_mcPlanType = spark.sql('''
select distinct beneID, state, max(jan) as jan, max(feb) as feb, max(mar) as mar, max(apr) as apr, max(may) as may, max(jun) as jun, max(jul) as jul, max(aug) as aug, max(sep) as sep, max(oct) as oct, max(nov) as nov, max(dec) as dec
FROM newtable 
GROUP BY beneID, state; 
''')

df_mcPlanType.show()

# COMMAND ----------

df_mcPlanType = df_mcPlanType.withColumnRenamed("jan","1").withColumnRenamed("feb","2").withColumnRenamed("mar","3").withColumnRenamed("apr","4").withColumnRenamed("may","5").withColumnRenamed("jun","6").withColumnRenamed("jul","7").withColumnRenamed("aug","8").withColumnRenamed("sep","9").withColumnRenamed("oct","10").withColumnRenamed("nov","11").withColumnRenamed("dec","12")
df_mcPlanType = df_mcPlanType.to_koalas().melt(id_vars=['beneID','state'], value_vars=['1','2','3','4','5','6','7','8','9','10','11','12']).to_spark().withColumnRenamed("variable", 'month').withColumnRenamed("value", 'mcPlanType')
df_mcPlanType.show(25)

# COMMAND ----------

print((df_mcPlanType.count(), len(df_mcPlanType.columns)))
df_mcPlanType = df_mcPlanType.dropDuplicates(["beneID","state","month"])
print((df_mcPlanType.count(), len(df_mcPlanType.columns)))

# COMMAND ----------

df_chipCode = df.select('BENE_ID','STATE_CD','CHIP_CD_01','CHIP_CD_02','CHIP_CD_03','CHIP_CD_04','CHIP_CD_05','CHIP_CD_06','CHIP_CD_07','CHIP_CD_08','CHIP_CD_09','CHIP_CD_10','CHIP_CD_11','CHIP_CD_12')

df_chipCode = df_chipCode.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", 'state').withColumnRenamed("CHIP_CD_01", 'jan').withColumnRenamed("CHIP_CD_02", 'feb').withColumnRenamed("CHIP_CD_03", 'mar').withColumnRenamed("CHIP_CD_04", 'apr').withColumnRenamed("CHIP_CD_05", 'may').withColumnRenamed("CHIP_CD_06", 'jun').withColumnRenamed("CHIP_CD_07", 'jul').withColumnRenamed("CHIP_CD_08", 'aug').withColumnRenamed("CHIP_CD_09", 'sep').withColumnRenamed("CHIP_CD_10", 'oct').withColumnRenamed("CHIP_CD_11", 'nov').withColumnRenamed("CHIP_CD_12", 'dec')

df_chipCode.show(25)

# COMMAND ----------

df_chipCode.registerTempTable("newtable")
df_chipCode = spark.sql('''
select distinct beneID, state, max(jan) as jan, max(feb) as feb, max(mar) as mar, max(apr) as apr, max(may) as may, max(jun) as jun, max(jul) as jul, max(aug) as aug, max(sep) as sep, max(oct) as oct, max(nov) as nov, max(dec) as dec
FROM newtable 
GROUP BY beneID, state; 
''')

df_chipCode.show()

# COMMAND ----------

df_chipCode = df_chipCode.withColumnRenamed("jan","1").withColumnRenamed("feb","2").withColumnRenamed("mar","3").withColumnRenamed("apr","4").withColumnRenamed("may","5").withColumnRenamed("jun","6").withColumnRenamed("jul","7").withColumnRenamed("aug","8").withColumnRenamed("sep","9").withColumnRenamed("oct","10").withColumnRenamed("nov","11").withColumnRenamed("dec","12")

df_chipCode = df_chipCode.to_koalas().melt(id_vars=['beneID','state'], value_vars=['1','2','3','4','5','6','7','8','9','10','11','12']).to_spark().withColumnRenamed("variable", 'month').withColumnRenamed("value", 'chipCode')
df_chipCode.show(20)

# COMMAND ----------

print((df_medDays.count(), len(df_medDays.columns)))
print((df_chipDays.count(), len(df_chipDays.columns)))
print((df_mcPlanType.count(), len(df_mcPlanType.columns)))
print((df_chipCode.count(), len(df_chipCode.columns)))
print((df_dual.count(), len(df_dual.columns)))

# COMMAND ----------

df_medDays= df_medDays.withColumn("medDaysInd", lit(1))
df_chipDays= df_chipDays.withColumn("chipDayInd", lit(1))
df_mcPlanType=df_mcPlanType.withColumn("mcPlanInd",lit(1))
df_chipCode=df_chipCode.withColumn("chipCodeInd",lit(1))
df_dual=df_dual.withColumn("dualInd",lit(1))

df_medDays.groupBy('medDaysInd').count().show()
df_chipDays.groupBy('chipDayInd').count().show()
df_mcPlanType.groupBy('mcPlanInd').count().show()
df_chipCode.groupBy('chipCodeInd').count().show()
df_dual.groupBy('dualInd').count().show()

# COMMAND ----------

print((df_medDays.count(), len(df_medDays.columns)))
df_medDays.show(1)

print((df_chipDays.count(), len(df_chipDays.columns)))
df_chipDays.show(1)

print((df_mcPlanType.count(), len(df_mcPlanType.columns)))
df_mcPlanType.show(1)

print((df_chipCode.count(), len(df_chipCode.columns)))
df_chipCode.show(1)

print((df_chipCode.count(), len(df_chipCode.columns)))
df_chipCode.show(1)

print((df_dual.count(), len(df_dual.columns)))
df_dual.show(1)

# COMMAND ----------

memberDf = df_medDays.join(df_chipDays, on=['beneID','state','month'], how='inner')
memberDf = memberDf.join(df_mcPlanType, on=['beneID','state','month'], how='inner')
memberDf = memberDf.join(df_chipCode, on=['beneID','state','month'], how='inner')
memberDf = memberDf.join(df_dual, on=['beneID','state','month'], how='inner')
print((memberDf.count(), len(memberDf.columns)))

# COMMAND ----------

memberDf.groupBy('medDaysInd').count().show()
memberDf.groupBy('chipDayInd').count().show()
memberDf.groupBy('mcPlanInd').count().show()
memberDf.groupBy('chipCodeInd').count().show()
memberDf.groupBy('dualInd').count().show()

# COMMAND ----------

#drop those three extra indicator variables
memberDf = memberDf.drop('medDaysInd','chipDayInd','mcPlanInd','chipCodeInd','dualInd')

# COMMAND ----------

test1 = memberDf.dropDuplicates(["beneID","state"])
print((test1.count(), len(test1.columns)))

test2 = spark.read.table("dua_058828_spa240.demo2019")
test2 = test2.dropDuplicates(["beneID","state"])
print((test2.count(), len(test2.columns)))

# COMMAND ----------

#code up indicator to identify medicaid vs. chip member
# > 0 days = enrollment
# 0 days or null = not enrolled
#remove members who are not chip or medicaid enrolled

memberDf = memberDf.withColumn("medEnroll", when((memberDf.medicaidDays>0), 'yes').when((memberDf.medicaidDays==0), 'no').otherwise('missing'))  
memberDf = memberDf.withColumn("chipEnroll", when((memberDf.chipDays>0), 'yes').when((memberDf.chipDays==0), 'no').otherwise('missing'))  
memberDf = memberDf.withColumn("enrolled", when(((memberDf.medEnroll=='yes') | (memberDf.medEnroll=='yes')), 'yes').otherwise('no'))  
memberDf.show(50)

memberDf.groupBy('medEnroll').count().orderBy(desc('count'),'medEnroll').show()
memberDf.groupBy('chipEnroll').count().orderBy(desc('count'),'chipEnroll').show()
memberDf.groupBy('enrolled').count().orderBy(desc('count'),'enrolled').show()

# COMMAND ----------

# flag patients who are dual medicare and medicaid eligibility
memberDf = memberDf.withColumn("dual", when((col("dualStatus").isin(['00'])), 'no').when((col("dualStatus").isin(['02','08','01','03','06','04','05','09','10'])), 'yes').otherwise('missing'))
memberDf.show(50)

memberDf.groupBy('dualStatus').count().orderBy(desc('count'),'dualStatus').show()
memberDf.groupBy('dual').count().orderBy(desc('count'),'dual').show()

# COMMAND ----------

memberDf = memberDf.withColumn("managedCare", when((col("mcPlanType").isin(['01','15','12','04','02','14','16','07','80','08','70','17','06','20','03','09','60'])), 'yes').otherwise('no'))
memberDf.show(50)

memberDf.groupBy('mcPlanType').count().orderBy(desc('count'),'mcPlanType').show()
memberDf.groupBy('managedCare').count().orderBy(desc('count'),'managedCare').show()

# COMMAND ----------

#code up chip code
memberDf = memberDf.withColumn("coverageType", when((col("chipCode").isin(['0'])), 'neither').when((col("chipCode").isin(['1'])), 'medicaid').when((col("chipCode").isin(['2','3'])), 'chip').when((col("chipCode").isin(['4'])), 'both').otherwise('missing'))
memberDf.show(50)

memberDf.groupBy('chipCode').count().orderBy(desc('count'),'chipCode').show()
memberDf.groupBy('coverageType').count().orderBy(desc('count'),'coverageType').show()

# COMMAND ----------

print(memberDf.printSchema())

# COMMAND ----------

#drop months with no coverage (enrolled="no")
#medEnroll,chipEnroll

memberDf = memberDf.where(memberDf.enrolled != 'no')
memberDf.groupBy('enrolled').count().orderBy(desc('count'),'enrolled').show()

#memberDf = memberDf.drop("medEnroll","chipEnroll") 
print(memberDf.printSchema())

# COMMAND ----------

memberDf.groupBy('enrolled').count().orderBy(desc('count'),'enrolled').show()
memberDf.groupBy('coverageType').count().orderBy(desc('count'),'coverageType').show()
memberDf.groupBy('managedCare').count().orderBy(desc('count'),'managedCare').show()
memberDf.groupBy('dual').count().orderBy(desc('count'),'dual').show()

# COMMAND ----------

print((memberDf.count(), len(memberDf.columns)))
memberDf.write.saveAsTable("dua_058828_spa240.elig2019", mode='overwrite')

# COMMAND ----------

#read demoFile ; create indicactor ; and merge into person-month file
#ensure size of DF is same pre- vs. post-merge
#validate no missing post-merge

demoFile = spark.read.table("dua_058828_spa240.demo2019")
demoFile=demoFile.withColumn("demoInd",lit(1))
print((memberDf.count(), len(memberDf.columns)))
memberDf = memberDf.join(demoFile, on=['beneID','state'], how='left')
print((memberDf.count(), len(memberDf.columns)))
memberDf.groupBy('demoInd').count().orderBy(desc('count'),'demoInd').show()

# COMMAND ----------

#drop unnecessary variables
memberDf = memberDf.drop('demoInd')
print(memberDf.printSchema())

# COMMAND ----------

print((memberDf.count(), len(memberDf.columns)))
memberDf.write.saveAsTable("dua_058828_spa240.member2019", mode='overwrite')

# COMMAND ----------

test = spark.table("dua_058828_spa240.member2019")
column_name = 'county'
null_count = test.filter(col(column_name).isNull()).count()
# Print the result
print("Total count of null values in column '{}': {}".format(column_name, null_count))

# COMMAND ----------

