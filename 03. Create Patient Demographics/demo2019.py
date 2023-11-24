# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count, sum

# COMMAND ----------

# Read member characteristics into pyspark
df=spark.sql("select BENE_ID, STATE_CD, BENE_STATE_CD, BENE_CNTY_CD, BENE_ZIP_CD, AGE, AGE_GRP_CD, DEATH_IND, SEX_CD, RACE_ETHNCTY_CD,ENGLSH_LANG_PRFCNCY_CD, MRTL_STUS_CD, HSEHLD_SIZE_CD, INCM_CD, CTZNSHP_IND, SSDI_IND, SSI_IND, TANF_CASH_CD, DSBLTY_DEAF_IND, DSBLTY_BLND_IND, DSBLTY_DFCLTY_CNCNTRTNG_IND, DSBLTY_DFCLTY_WLKG_IND, DSBLTY_DFCLTY_DRSNG_BATHNG_IND, DSBLTY_DFCLTY_ERNDS_IND, DSBLTY_OTHR_IND from dua_058828.tafr19_demog_elig_base")

# COMMAND ----------

print((df.count(), len(df.columns)))
df = df.dropna(subset=["BENE_ID",'STATE_CD'])
print((df.count(), len(df.columns)))

# COMMAND ----------

# Create a new column 'repair' based on the value of 'rep78'
df = df.withColumn("ageCat", when((df.AGE<10), 'under10').when((9<df.AGE) & (df.AGE<18), '10To17').when((17<df.AGE) & (df.AGE<30), '18To29').when((29<df.AGE) & (df.AGE<40), '30To39').when((39<df.AGE) & (df.AGE<50), '40To49').when((49<df.AGE) & (df.AGE<65), '50To64').when((df.AGE>64), 'over64').otherwise('missing'))      

# COMMAND ----------

df.groupBy('ageCat').count().orderBy(desc('count'),'ageCat').show()
df.groupBy('AGE_GRP_CD').count().orderBy(desc('count'),'AGE_GRP_CD').show()

# COMMAND ----------

df = df.withColumn("death", 
                        when((col("DEATH_IND")== '0'), 'no')
                       .when((col("DEATH_IND")== '1'), 'yes')
                        .otherwise('missing'))

# COMMAND ----------

df.groupBy('death').count().orderBy(desc('count'),'death').show()
df.groupBy('DEATH_IND').count().orderBy(desc('count'),'DEATH_IND').show()

# COMMAND ----------

df = df.withColumn("sex", 
                        when((col("SEX_CD")== 'M'), 'male')
                       .when((col("SEX_CD")== 'F'), 'female')
                        .otherwise('missing'))

# COMMAND ----------

df.groupBy('sex').count().orderBy(desc('count'),'sex').show()
df.groupBy('SEX_CD').count().orderBy(desc('count'),'SEX_CD').show()

# COMMAND ----------

df = df.withColumn("race", 
                        when((col("RACE_ETHNCTY_CD")== 1), 'white')
                       .when((col("RACE_ETHNCTY_CD")== 2), 'black')
                       .when((col("RACE_ETHNCTY_CD")== 3), 'asian')
                       .when((col("RACE_ETHNCTY_CD")== 4), 'native')
                       .when((col("RACE_ETHNCTY_CD")== 5), 'hawaiian')
                       .when((col("RACE_ETHNCTY_CD")== 6), 'multiracial')
                       .when((col("RACE_ETHNCTY_CD")== 7), 'hispanic')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('race').count().orderBy(desc('count'),'race').show()
df.groupBy('RACE_ETHNCTY_CD').count().orderBy(desc('count'),'RACE_ETHNCTY_CD').show()

# COMMAND ----------

df = df.withColumn("speakEnglish", 
                        when((col("ENGLSH_LANG_PRFCNCY_CD").isin([0,1])), 'yes')
                       .when((col("ENGLSH_LANG_PRFCNCY_CD").isin([2,3])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('speakEnglish').count().orderBy(desc('count'),'speakEnglish').show()
df.groupBy('ENGLSH_LANG_PRFCNCY_CD').count().orderBy(desc('count'),'ENGLSH_LANG_PRFCNCY_CD').show()

# COMMAND ----------

df = df.withColumn("married", 
                        when((col("MRTL_STUS_CD").isin(['01','02','03','04','05','06','07','08'])), 'yes')
                       .when((col("MRTL_STUS_CD").isin(['09','10','11','12','13','14'])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('married').count().orderBy(desc('count'),'married').show()
df.groupBy('MRTL_STUS_CD').count().orderBy(desc('count'),'MRTL_STUS_CD').show()

# COMMAND ----------

df = df.withColumn("houseSize", 
                        when((col("HSEHLD_SIZE_CD").isin(['01'])), 'single')
                       .when((col("HSEHLD_SIZE_CD").isin(['02','03','04','05'])), 'twoToFive')
                       .when((col("HSEHLD_SIZE_CD").isin(['06','07','08'])), 'sixorMore')                   
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('houseSize').count().orderBy(desc('count'),'houseSize').show()
df.groupBy('HSEHLD_SIZE_CD').count().orderBy(desc('count'),'HSEHLD_SIZE_CD').show()

# COMMAND ----------

df = df.withColumn("fedPovLine", 
                        when((col("INCM_CD").isin(['01'])), '0To100')
                       .when((col("INCM_CD").isin(['02','03','04'])), '100To200')
                       .when((col("INCM_CD").isin(['05','06','07','08'])), '200AndMore')                   
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('fedPovLine').count().orderBy(desc('count'),'fedPovLine').show()
df.groupBy('INCM_CD').count().orderBy(desc('count'),'INCM_CD').show()

# COMMAND ----------

df = df.withColumn("UsCitizen", 
                        when((col("CTZNSHP_IND").isin(['1'])), 'yes')
                       .when((col("CTZNSHP_IND").isin(['0'])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('UsCitizen').count().orderBy(desc('count'),'UsCitizen').show()
df.groupBy('CTZNSHP_IND').count().orderBy(desc('count'),'CTZNSHP_IND').show()

# COMMAND ----------

df = df.withColumn("ssdi", 
                        when((col("SSDI_IND").isin(['1'])), 'yes')
                       .when((col("SSDI_IND").isin(['0'])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('ssdi').count().orderBy(desc('count'),'ssdi').show()
df.groupBy('SSDI_IND').count().orderBy(desc('count'),'SSDI_IND').show()

# COMMAND ----------

df = df.withColumn("ssi", 
                        when((col("SSI_IND").isin(['1'])), 'yes')
                       .when((col("SSI_IND").isin(['0'])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('ssi').count().orderBy(desc('count'),'ssi').show()
df.groupBy('SSI_IND').count().orderBy(desc('count'),'SSI_IND').show()

# COMMAND ----------

df = df.withColumn("tanf", 
                        when((col("TANF_CASH_CD").isin([2])), 'yes')
                       .when((col("TANF_CASH_CD").isin([0,1])), 'no')
                       .otherwise('missing'))

# COMMAND ----------

df.groupBy('tanf').count().orderBy(desc('count'),'tanf').show()
df.groupBy('TANF_CASH_CD').count().orderBy(desc('count'),'TANF_CASH_CD').show()

# COMMAND ----------

#measure indicators for disability
df.groupby(['DSBLTY_DEAF_IND','DSBLTY_BLND_IND','DSBLTY_DFCLTY_CNCNTRTNG_IND','DSBLTY_DFCLTY_WLKG_IND','DSBLTY_DFCLTY_DRSNG_BATHNG_IND','DSBLTY_DFCLTY_ERNDS_IND','DSBLTY_OTHR_IND']).count().show()

# COMMAND ----------

#recode disabled variable as 0/1 
df = df.withColumn("disabled", F.when((df.DSBLTY_DEAF_IND == 1.0) | (df.DSBLTY_BLND_IND == 1.0) | (df.DSBLTY_DFCLTY_CNCNTRTNG_IND == 1.0) | (df.DSBLTY_DFCLTY_WLKG_IND == 1.0) | (df.DSBLTY_DFCLTY_DRSNG_BATHNG_IND == 1.0) | (df.DSBLTY_DFCLTY_ERNDS_IND == 1.0) | (df.DSBLTY_OTHR_IND == 1.0), 'yes').when((df.DSBLTY_DEAF_IND == 0.0) & (df.DSBLTY_BLND_IND == 0.0) & (df.DSBLTY_DFCLTY_CNCNTRTNG_IND == 0.0) & (df.DSBLTY_DFCLTY_WLKG_IND == 0.0) & (df.DSBLTY_DFCLTY_DRSNG_BATHNG_IND ==0.0) & (df.DSBLTY_DFCLTY_ERNDS_IND == 0.0) & (df.DSBLTY_OTHR_IND == 0.0), 'no').otherwise('missing'))

# COMMAND ----------

df.groupBy('disabled').count().orderBy(desc('count'),'disabled').show()

# COMMAND ----------

#drop columns
columns_to_drop = ['AGE_GRP_CD','DEATH_IND','SEX_CD','RACE_ETHNCTY_CD','ENGLSH_LANG_PRFCNCY_CD','MRTL_STUS_CD','HSEHLD_SIZE_CD','INCM_CD', 'CTZNSHP_IND','SSDI_IND','SSI_IND','TANF_CASH_CD','DSBLTY_DEAF_IND','DSBLTY_BLND_IND','DSBLTY_DFCLTY_CNCNTRTNG_IND','DSBLTY_DFCLTY_WLKG_IND','DSBLTY_DFCLTY_DRSNG_BATHNG_IND','DSBLTY_DFCLTY_ERNDS_IND','DSBLTY_OTHR_IND']
df = df.drop(*columns_to_drop)

#rename columns
df=df.withColumnRenamed("BENE_ID","beneID").withColumnRenamed("STATE_CD","state").withColumnRenamed("BENE_STATE_CD","stateFips").withColumnRenamed("BENE_CNTY_CD","countyFips").withColumnRenamed("BENE_ZIP_CD","zipCode").withColumnRenamed("AGE","age")

# COMMAND ----------

#verify coding was accurate
from pyspark.sql.functions import col, count, desc

df.groupby('race').count().orderBy(desc('count'),'race').show()
df.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()
df.groupby('sex').count().orderBy(desc('count'),'sex').show()
df.groupby('race').count().orderBy(desc('count'),'race').show()
df.groupby('death').count().orderBy(desc('count'),'death').show()
df.groupby('speakEnglish').count().orderBy(desc('count'),'speakEnglish').show()
df.groupby('married').count().orderBy(desc('count'),'married').show()
df.groupby('houseSize').count().orderBy(desc('count'),'houseSize').show()
df.groupby('fedPovLine').count().orderBy(desc('count'),'fedPovLine').show()
df.groupby('UsCitizen').count().orderBy(desc('count'),'UsCitizen').show()
df.groupby('ssdi').count().orderBy(desc('count'),'ssdi').show()
df.groupby('ssi').count().orderBy(desc('count'),'ssi').show()
df.groupby('tanf').count().orderBy(desc('count'),'tanf').show()
df.groupby('disabled').count().orderBy(desc('count'),'disabled').show()

# COMMAND ----------

# Use the concat() function to concatenate the two columns into a new column called "combined_message"
df = df.withColumn("county", concat("stateFips", "countyFips"))

# COMMAND ----------

#measure missingness for these features
df_Columns=["beneID","state","stateFips","countyFips","county","zipCode","age","ageCat","sex","race","death","speakEnglish","married","houseSize","fedPovLine","UsCitizen","ssdi","ssi","tanf","disabled"]
df2 = df.select([count(when(col(c).contains('None') | \
                            col(c).contains('NULL') | \
                            (col(c) == '' ) | \
                            col(c).isNull() | \
                            isnan(c), c 
                           )).alias(c)
                    for c in df_Columns])
df2.show()

# COMMAND ----------

df.show(25)

# COMMAND ----------

columns_to_drop = ['stateFips','countyFips']
df = df.drop(*columns_to_drop)
df.show()

# COMMAND ----------

df.registerTempTable("connections")

# COMMAND ----------

dfAgg = spark.sql('''
SELECT beneID, state, max(county) as county, max(zipCode) as zipCode, max(age) as age, max(ageCat) as ageCat, max(sex) as sex, max(race) as race, max(death) as death, max(speakEnglish) as speakEnglish, max(married) as married, max(houseSize) as houseSize, max(fedPovLine) as fedPovLine, max(UsCitizen) as UsCitizen, max(ssdi) as ssdi, max(ssi) as ssi, max(tanf) as tanf, max(disabled) as disabled 
FROM connections
GROUP BY beneID, state;
''')

# COMMAND ----------

print((df.count(), len(df.columns)))
print((dfAgg.count(), len(dfAgg.columns)))

# COMMAND ----------

df.groupby('race').count().orderBy(desc('count'),'race').show()
dfAgg.groupby('race').count().orderBy(desc('count'),'race').show()

df.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()
dfAgg.groupby('ageCat').count().orderBy(desc('count'),'ageCat').show()

df.groupby('sex').count().orderBy(desc('count'),'sex').show()
dfAgg.groupby('sex').count().orderBy(desc('count'),'sex').show()

df.groupby('death').count().orderBy(desc('count'),'death').show()
dfAgg.groupby('death').count().orderBy(desc('count'),'death').show()

dfAgg.groupby('speakEnglish').count().orderBy(desc('count'),'speakEnglish').show()
dfAgg.groupby('married').count().orderBy(desc('count'),'married').show()
dfAgg.groupby('houseSize').count().orderBy(desc('count'),'houseSize').show()
dfAgg.groupby('fedPovLine').count().orderBy(desc('count'),'fedPovLine').show()
dfAgg.groupby('UsCitizen').count().orderBy(desc('count'),'UsCitizen').show()
dfAgg.groupby('ssdi').count().orderBy(desc('count'),'ssdi').show()
dfAgg.groupby('ssi').count().orderBy(desc('count'),'ssi').show()
dfAgg.groupby('tanf').count().orderBy(desc('count'),'tanf').show()
dfAgg.groupby('disabled').count().orderBy(desc('count'),'disabled').show()

# COMMAND ----------

print(dfAgg.printSchema())

# COMMAND ----------

dfAgg.write.saveAsTable("dua_058828_spa240.demo2019", mode='overwrite')

# COMMAND ----------

dfA = spark.read.table("dua_058828_spa240.demo2019")
print((dfA.count(), len(dfA.columns)))
dfA = dfA.dropDuplicates(["beneID","state"])
dfA = dfA.select("beneID","state")
print((dfA.count(), len(dfA.columns)))
dfA.show()