# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

df = spark.table("dua_058828_spa240.demo2017")
print((df.count(), len(df.columns)))

# COMMAND ----------



# COMMAND ----------

df = spark.table("dua_058828_spa240.demo2017")
print((df.count(), len(df.columns)))

df = df.dropDuplicates(['beneID'])
print((df.count(), len(df.columns)))

df = df.filter(df.state.isin(['AL','AZ','DC','DE','HI','ID','IL','IN','KS','KY','LA','MD','ME','MI','MS','MT','ND','NM','NV','PA','TN','UT','VA','VT','WA','WV','WY']))
print((df.count(), len(df.columns)))

df = df.withColumn("censusRegion", 
                        when((col("state").isin(['AZ','HI','ID','MT','NM','NV','UT','WA','WY'])), 'West')
                       .when((col("state").isin(['IL','IN','KS','MI','ND'])), 'Midwest')
                       .when((col("state").isin(['AL','KY','LA','MS','TN','WV','VA','MD','DC'])), 'South')                   
                       .otherwise('Northeast'))

df.groupBy('censusRegion').count().orderBy(desc('count'),'censusRegion').show()

memberDf = spark.table("dua_058828_spa240.elig2017")
memberDf = memberDf.select("beneID", "state", "month", "dual", "enrolled","managedCare","medicaidDays","medEnroll")
memberDf = memberDf.withColumn("dualInd", when((col("dual").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("enrolledInd", when((col("enrolled").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("medicaidMonths", when((col("medEnroll").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("managedCareMonths", when((col("managedCare").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("janElig", when((col("month").isin(['1'])), lit(1)).otherwise(lit(0))).withColumn("febElig", when((col("month").isin(['2'])), lit(1)).otherwise(lit(0))).withColumn("marElig", when((col("month").isin(['3'])), lit(1)).otherwise(lit(0))).withColumn("aprElig", when((col("month").isin(['4'])), lit(1)).otherwise(lit(0))).withColumn("mayElig", when((col("month").isin(['5'])), lit(1)).otherwise(lit(0))).withColumn("junElig", when((col("month").isin(['6'])), lit(1)).otherwise(lit(0))).withColumn("julElig", when((col("month").isin(['7'])), lit(1)).otherwise(lit(0))).withColumn("augElig", when((col("month").isin(['8'])), lit(1)).otherwise(lit(0))).withColumn("sepElig", when((col("month").isin(['9'])), lit(1)).otherwise(lit(0))).withColumn("octElig", when((col("month").isin(['10'])), lit(1)).otherwise(lit(0))).withColumn("novElig", when((col("month").isin(['11'])), lit(1)).otherwise(lit(0))).withColumn("decElig", when((col("month").isin(['12'])), lit(1)).otherwise(lit(0)))

memberDf.registerTempTable("connections")
memberDf = spark.sql('''
SELECT beneID, state, sum(enrolledInd) as enrolledMonths, sum(dualInd) as medicareMonths, sum(medicaidDays) as medicaidDays, sum(medicaidMonths) as medicaidMonths, sum(managedCareMonths) as managedCareMonths, 
max(janElig) as janElig, max(febElig) as febElig, max(marElig) as marElig, max(aprElig) as aprElig, max(mayElig) as mayElig, max(junElig) as junElig, 
max(julElig) as julElig, max(augElig) as augElig, max(sepElig) as sepElig, max(octElig) as octElig, max(novElig) as novElig, max(decElig) as decElig
FROM connections
GROUP BY beneID, state;
''')

dfNew = df.join(memberDf, on=['beneID','state'], how='left')

dfNew = dfNew.filter(dfNew.medicareMonths ==0)
dfNew = dfNew.filter(dfNew.ageCat != 'over64')
dfNew.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()                  

print((dfNew.count(), len(dfNew.columns)))

dfNew.write.saveAsTable("dua_058828_spa240.finalSample2017", mode='overwrite')

dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()

dfNew = dfNew.filter(dfNew.enrolledMonths ==12)
print((dfNew.count(), len(dfNew.columns)))

dfNew.groupby('medicaidMonths').count().orderBy(desc('count'),'medicaidMonths').show()
dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()
dfNew.groupby('managedCareMonths').count().orderBy(desc('count'),'managedCareMonths').show()

summary = dfNew.select(['medicaidDays']).describe()
summary.show()

dfNew.write.saveAsTable("dua_058828_spa240.enrolledSample2017", mode='overwrite')

# COMMAND ----------

memberDf = spark.table("dua_058828_spa240.finalSample2017")

memberDf.groupby('janElig').count().orderBy(desc('count'),'janElig').show()
memberDf.groupby('febElig').count().orderBy(desc('count'),'febElig').show()
memberDf.groupby('marElig').count().orderBy(desc('count'),'marElig').show()
memberDf.groupby('aprElig').count().orderBy(desc('count'),'aprElig').show()
memberDf.groupby('mayElig').count().orderBy(desc('count'),'mayElig').show()
memberDf.groupby('junElig').count().orderBy(desc('count'),'junElig').show()

memberDf.groupby('julElig').count().orderBy(desc('count'),'julElig').show()
memberDf.groupby('augElig').count().orderBy(desc('count'),'augElig').show()
memberDf.groupby('sepElig').count().orderBy(desc('count'),'sepElig').show()
memberDf.groupby('octElig').count().orderBy(desc('count'),'octElig').show()
memberDf.groupby('novElig').count().orderBy(desc('count'),'novElig').show()
memberDf.groupby('decElig').count().orderBy(desc('count'),'decElig').show()

#memberDf.show(100)

# COMMAND ----------

df = spark.table("dua_058828_spa240.demo2018")
print((df.count(), len(df.columns)))

df = df.dropDuplicates(['beneID'])
print((df.count(), len(df.columns)))

df = df.filter(df.state.isin(['AL','AZ','DC','DE','HI','ID','IL','IN','KS','KY','LA','MD','ME','MI','MS','MT','ND','NM','NV','PA','TN','UT','VA','VT','WA','WV','WY']))
print((df.count(), len(df.columns)))

df = df.withColumn("censusRegion", 
                        when((col("state").isin(['AZ','HI','ID','MT','NM','NV','UT','WA','WY'])), 'West')
                       .when((col("state").isin(['IL','IN','KS','MI','ND'])), 'Midwest')
                       .when((col("state").isin(['AL','KY','LA','MS','TN','WV','VA','MD','DC'])), 'South')                   
                       .otherwise('Northeast'))

memberDf = spark.table("dua_058828_spa240.elig2018")
memberDf = memberDf.select("beneID", "state", "month", "dual", "enrolled","managedCare","medicaidDays","medEnroll")
memberDf = memberDf.withColumn("dualInd", when((col("dual").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("enrolledInd", when((col("enrolled").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("medicaidMonths", when((col("medEnroll").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("managedCareMonths", when((col("managedCare").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("janElig", when((col("month").isin(['1'])), lit(1)).otherwise(lit(0))).withColumn("febElig", when((col("month").isin(['2'])), lit(1)).otherwise(lit(0))).withColumn("marElig", when((col("month").isin(['3'])), lit(1)).otherwise(lit(0))).withColumn("aprElig", when((col("month").isin(['4'])), lit(1)).otherwise(lit(0))).withColumn("mayElig", when((col("month").isin(['5'])), lit(1)).otherwise(lit(0))).withColumn("junElig", when((col("month").isin(['6'])), lit(1)).otherwise(lit(0))).withColumn("julElig", when((col("month").isin(['7'])), lit(1)).otherwise(lit(0))).withColumn("augElig", when((col("month").isin(['8'])), lit(1)).otherwise(lit(0))).withColumn("sepElig", when((col("month").isin(['9'])), lit(1)).otherwise(lit(0))).withColumn("octElig", when((col("month").isin(['10'])), lit(1)).otherwise(lit(0))).withColumn("novElig", when((col("month").isin(['11'])), lit(1)).otherwise(lit(0))).withColumn("decElig", when((col("month").isin(['12'])), lit(1)).otherwise(lit(0)))

memberDf.registerTempTable("connections")
memberDf = spark.sql('''
SELECT beneID, state, sum(enrolledInd) as enrolledMonths, sum(dualInd) as medicareMonths, sum(medicaidDays) as medicaidDays, sum(medicaidMonths) as medicaidMonths, sum(managedCareMonths) as managedCareMonths, 
max(janElig) as janElig, max(febElig) as febElig, max(marElig) as marElig, max(aprElig) as aprElig, max(mayElig) as mayElig, max(junElig) as junElig, 
max(julElig) as julElig, max(augElig) as augElig, max(sepElig) as sepElig, max(octElig) as octElig, max(novElig) as novElig, max(decElig) as decElig
FROM connections
GROUP BY beneID, state;
''')

dfNew = df.join(memberDf, on=['beneID','state'], how='left')

dfNew = dfNew.filter(dfNew.medicareMonths ==0)
dfNew = dfNew.filter(dfNew.ageCat != 'over64')
dfNew.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()        
                     
print((dfNew.count(), len(dfNew.columns)))

dfNew.write.saveAsTable("dua_058828_spa240.finalSample2018", mode='overwrite')

dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()

dfNew = dfNew.filter(dfNew.enrolledMonths ==12)
print((dfNew.count(), len(dfNew.columns)))

dfNew.groupby('medicaidMonths').count().orderBy(desc('count'),'medicaidMonths').show()
dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()
dfNew.groupby('managedCareMonths').count().orderBy(desc('count'),'managedCareMonths').show()

summary = dfNew.select(['medicaidDays']).describe()
summary.show()

dfNew.write.saveAsTable("dua_058828_spa240.enrolledSample2018", mode='overwrite')

# COMMAND ----------

memberDf = spark.table("dua_058828_spa240.finalSample2018")

memberDf.groupby('janElig').count().orderBy(desc('count'),'janElig').show()
memberDf.groupby('febElig').count().orderBy(desc('count'),'febElig').show()
memberDf.groupby('marElig').count().orderBy(desc('count'),'marElig').show()
memberDf.groupby('aprElig').count().orderBy(desc('count'),'aprElig').show()
memberDf.groupby('mayElig').count().orderBy(desc('count'),'mayElig').show()
memberDf.groupby('junElig').count().orderBy(desc('count'),'junElig').show()

memberDf.groupby('julElig').count().orderBy(desc('count'),'julElig').show()
memberDf.groupby('augElig').count().orderBy(desc('count'),'augElig').show()
memberDf.groupby('sepElig').count().orderBy(desc('count'),'sepElig').show()
memberDf.groupby('octElig').count().orderBy(desc('count'),'octElig').show()
memberDf.groupby('novElig').count().orderBy(desc('count'),'novElig').show()
memberDf.groupby('decElig').count().orderBy(desc('count'),'decElig').show()

#memberDf.show(100)

# COMMAND ----------

df = spark.table("dua_058828_spa240.demo2019")
print((df.count(), len(df.columns)))

df = df.dropDuplicates(['beneID'])
print((df.count(), len(df.columns)))

df = df.filter(df.state.isin(['AL','AZ','DC','DE','HI','ID','IL','IN','KS','KY','LA','MD','ME','MI','MS','MT','ND','NM','NV','PA','TN','UT','VA','VT','WA','WV','WY']))
print((df.count(), len(df.columns)))

df = df.withColumn("censusRegion", 
                        when((col("state").isin(['AZ','HI','ID','MT','NM','NV','UT','WA','WY'])), 'West')
                       .when((col("state").isin(['IL','IN','KS','MI','ND'])), 'Midwest')
                       .when((col("state").isin(['AL','KY','LA','MS','TN','WV','VA','MD','DC'])), 'South')                   
                       .otherwise('Northeast'))

memberDf = spark.table("dua_058828_spa240.elig2019")
memberDf = memberDf.select("beneID", "state", "month", "dual", "enrolled","managedCare","medicaidDays","medEnroll")
memberDf = memberDf.withColumn("dualInd", when((col("dual").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("enrolledInd", when((col("enrolled").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("medicaidMonths", when((col("medEnroll").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("managedCareMonths", when((col("managedCare").isin(['yes'])), lit(1)).otherwise(lit(0))).withColumn("janElig", when((col("month").isin(['1'])), lit(1)).otherwise(lit(0))).withColumn("febElig", when((col("month").isin(['2'])), lit(1)).otherwise(lit(0))).withColumn("marElig", when((col("month").isin(['3'])), lit(1)).otherwise(lit(0))).withColumn("aprElig", when((col("month").isin(['4'])), lit(1)).otherwise(lit(0))).withColumn("mayElig", when((col("month").isin(['5'])), lit(1)).otherwise(lit(0))).withColumn("junElig", when((col("month").isin(['6'])), lit(1)).otherwise(lit(0))).withColumn("julElig", when((col("month").isin(['7'])), lit(1)).otherwise(lit(0))).withColumn("augElig", when((col("month").isin(['8'])), lit(1)).otherwise(lit(0))).withColumn("sepElig", when((col("month").isin(['9'])), lit(1)).otherwise(lit(0))).withColumn("octElig", when((col("month").isin(['10'])), lit(1)).otherwise(lit(0))).withColumn("novElig", when((col("month").isin(['11'])), lit(1)).otherwise(lit(0))).withColumn("decElig", when((col("month").isin(['12'])), lit(1)).otherwise(lit(0)))

memberDf.registerTempTable("connections")
memberDf = spark.sql('''
SELECT beneID, state, sum(enrolledInd) as enrolledMonths, sum(dualInd) as medicareMonths, sum(medicaidDays) as medicaidDays, sum(medicaidMonths) as medicaidMonths, sum(managedCareMonths) as managedCareMonths, 
max(janElig) as janElig, max(febElig) as febElig, max(marElig) as marElig, max(aprElig) as aprElig, max(mayElig) as mayElig, max(junElig) as junElig, 
max(julElig) as julElig, max(augElig) as augElig, max(sepElig) as sepElig, max(octElig) as octElig, max(novElig) as novElig, max(decElig) as decElig
FROM connections
GROUP BY beneID, state;
''')

dfNew = df.join(memberDf, on=['beneID','state'], how='left')

dfNew = dfNew.filter(dfNew.medicareMonths ==0)
dfNew = dfNew.filter(dfNew.ageCat != 'over64')
dfNew.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()   

print((dfNew.count(), len(dfNew.columns)))

dfNew.write.saveAsTable("dua_058828_spa240.finalSample2019", mode='overwrite')

dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()

dfNew = dfNew.filter(dfNew.enrolledMonths ==12)
print((dfNew.count(), len(dfNew.columns)))

dfNew.groupby('medicaidMonths').count().orderBy(desc('count'),'medicaidMonths').show()
dfNew.groupby('enrolledMonths').count().orderBy(desc('count'),'enrolledMonths').show()
dfNew.groupby('managedCareMonths').count().orderBy(desc('count'),'managedCareMonths').show()

summary = dfNew.select(['medicaidDays']).describe()
summary.show()

dfNew.write.saveAsTable("dua_058828_spa240.enrolledSample2019", mode='overwrite') 

# COMMAND ----------

memberDf = spark.table("dua_058828_spa240.finalSample2019")

memberDf.groupby('janElig').count().orderBy(desc('count'),'janElig').show()
memberDf.groupby('febElig').count().orderBy(desc('count'),'febElig').show()
memberDf.groupby('marElig').count().orderBy(desc('count'),'marElig').show()
memberDf.groupby('aprElig').count().orderBy(desc('count'),'aprElig').show()
memberDf.groupby('mayElig').count().orderBy(desc('count'),'mayElig').show()
memberDf.groupby('junElig').count().orderBy(desc('count'),'junElig').show()

memberDf.groupby('julElig').count().orderBy(desc('count'),'julElig').show()
memberDf.groupby('augElig').count().orderBy(desc('count'),'augElig').show()
memberDf.groupby('sepElig').count().orderBy(desc('count'),'sepElig').show()
memberDf.groupby('octElig').count().orderBy(desc('count'),'octElig').show()
memberDf.groupby('novElig').count().orderBy(desc('count'),'novElig').show()
memberDf.groupby('decElig').count().orderBy(desc('count'),'decElig').show()

#memberDf.show(100)