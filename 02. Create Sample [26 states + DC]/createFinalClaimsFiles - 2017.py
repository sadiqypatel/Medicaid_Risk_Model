# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

#member
member = spark.table("dua_058828_spa240.finalSample2017")
#print((df.count(), len(df.columns)))

member.registerTempTable("connections")
member = spark.sql('''
SELECT distinct beneID, state
FROM connections;
''')

# COMMAND ----------

#claims
# other = spark.table("dua_058828_spa240.otherservices2017")
# print((other.count(), len(other.columns)))
# other = other.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", "state")


# other = other.join(member, on=['beneID','state'], how='left')
# print((other.count(), len(other.columns)))

# other.write.saveAsTable("dua_058828_spa240.otherservices_final2017", mode='overwrite')

# COMMAND ----------

#inpatient
inpatient = spark.table("dua_058828_spa240.inpatient2017")
print((inpatient.count(), len(inpatient.columns)))
inpatient = inpatient.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", "state")

inpatient = inpatient.join(member, on=['beneID','state'], how='left')
print((inpatient.count(), len(inpatient.columns)))

inpatient.write.saveAsTable("dua_058828_spa240.inpatient_final2017", mode='overwrite')

# COMMAND ----------

#longTerm
longTerm = spark.table("dua_058828_spa240.longterm2017")
print((longTerm.count(), len(longTerm.columns)))
longTerm = longTerm.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", "state")

longTerm = longTerm.join(member, on=['beneID','state'], how='left')
print((longTerm.count(), len(longTerm.columns)))

longTerm.write.saveAsTable("dua_058828_spa240.longterm_final2017", mode='overwrite')

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum

#member
member = spark.table("dua_058828_spa240.finalSample2017")
#print((df.count(), len(df.columns)))

member.registerTempTable("connections")
member = spark.sql('''
SELECT distinct beneID, state
FROM connections;
''')

#pharm
pharm = spark.table("dua_058828_spa240.pharm2017")
print((pharm.count(), len(pharm.columns)))
pharm = pharm.withColumnRenamed("BENE_ID", "beneID").withColumnRenamed("STATE_CD", "state")

pharm = pharm.join(member, on=['beneID','state'], how='inner')
print((pharm.count(), len(pharm.columns)))

pharm.write.saveAsTable("dua_058828_spa240.pharm_final2017", mode='overwrite')

# COMMAND ----------

