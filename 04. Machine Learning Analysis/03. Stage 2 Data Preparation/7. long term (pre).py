# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

# COMMAND ----------

longterm = spark.table("dua_058828_spa240.paper1_stage2_longterm_12month")

print((longterm.count(), len(longterm.columns)))
longterm = longterm.filter(longterm.pre==1)
print((longterm.count(), len(longterm.columns)))
longterm = longterm.select("beneID","state","CLM_ID","SRVC_BGN_DT", "SRVC_END_DT")
print(longterm.printSchema())

# COMMAND ----------

df =  longterm
df = df.withColumnRenamed("SRVC_BGN_DT", "StartDate").withColumnRenamed("SRVC_END_DT", "EndDate")
print(df.printSchema())

# COMMAND ----------

df = df.withColumn("StartDate", col("StartDate").cast("date"))
df = df.withColumn("EndDate", col("EndDate").cast("date"))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, sum as cumsum, when, row_number
from pyspark.sql.window import Window

def episodesOfCare(df):
    # Define window specifications for calculating lag values, cumulative sum, and row number
    beneID_state_window = Window.partitionBy("beneID", "state").orderBy("StartDate", "EndDate")
    beneID_state_window_cumsum = Window.partitionBy("beneID", "state").orderBy("StartDate", "EndDate").rowsBetween(Window.unboundedPreceding, 0)

    # Calculate lag values for StartDate and EndDate columns
    df = df.withColumn("prev_StartDate", lag("StartDate").over(beneID_state_window))
    df = df.withColumn("prev_EndDate", lag("EndDate").over(beneID_state_window))

    # Calculate row number within each group
    df = df.withColumn("row_num", row_number().over(beneID_state_window))

    # Define conditions for new episode and overlap types
    new_episode_condition = (col("StartDate") > col("prev_EndDate") + 1) | col("prev_EndDate").isNull()
    regular_overlap_condition = (col("StartDate") <= col("prev_EndDate") + 1) & (col("EndDate") > col("prev_EndDate"))
    same_start_date_condition = (col("StartDate") == col("prev_StartDate")) & (col("EndDate") < col("prev_EndDate"))
    embedded_condition = (col("StartDate") > col("prev_StartDate")) & (col("EndDate") < col("prev_EndDate"))
    perfect_overlap_condition = (col("StartDate") == col("prev_StartDate")) & (col("EndDate") == col("prev_EndDate"))

    # Assign new episode flag based on condition
    df = df.withColumn("new_episode_flag", new_episode_condition.cast("int"))

    # Calculate episode numbers using cumulative sum
    df = df.withColumn("episode", cumsum("new_episode_flag").over(beneID_state_window_cumsum))

    df = df.withColumn("ovlp", 
                   when(col("row_num") == 1, "1.First")
                   .when(new_episode_condition, "2.New Episode")
                   .when(regular_overlap_condition, "3.Regular Overlap")
                   .when(same_start_date_condition, "5.Same Start Date (SRO)")
                   .when(embedded_condition, "6.Embedded")
                   .when(perfect_overlap_condition, "7.Perfect Overlap"))

    # Drop unnecessary columns
    df = df.drop("prev_StartDate", "prev_EndDate", "new_episode_flag", "row_num")

    return df
  
# Convert 'StartDate' and 'EndDate' columns to date type
df = df.withColumn("StartDate", col("StartDate").cast("date"))
df = df.withColumn("EndDate", col("EndDate").cast("date"))

# Apply the episodesOfCare function
result_df = episodesOfCare(df)

# Sort the DataFrame by beneID, state, StartDate, and EndDate
#result_df = result_df.orderBy("beneID", "state", "StartDate", "EndDate")

# Show the result
result_df.show(1000)

# COMMAND ----------

result_df.registerTempTable("connections")

long_term_episode = spark.sql('''
SELECT beneID, state, episode, min(StartDate) as StartDate, max(EndDate) as EndDate
FROM connections
GROUP BY beneID, state, episode;
''')

long_term_episode.show(100)

# COMMAND ----------

from pyspark.sql.functions import col, when, datediff

# Calculate inpatient days based on "all_cause_ip" column
long_term_episode = long_term_episode.withColumn("long_term_days", datediff(col("EndDate"), col("StartDate")))
long_term_episode = long_term_episode.withColumn("long_term_episodes", lit(1))
long_term_episode.show()

# COMMAND ----------

long_term_episode.registerTempTable("connections")

long_term_total = spark.sql('''
SELECT beneID, state, sum(long_term_episodes) as long_term_episodes, sum(long_term_days) as long_term_days
FROM connections
GROUP BY beneID, state;
''')

long_term_total.show(100)

# COMMAND ----------

member = spark.table("dua_058828_spa240.demographic_stage2")
member = member.select("beneID", "state")
print((member.count(), len(member.columns)))
print(member.printSchema())

from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf

# # Left join 'df' with 'ed' based on the 'DGNS_CD_1' column
long_term_predictors = member.join(long_term_total, on=['beneID','state'], how='left').fillna(0)

long_term_predictors.show(200)

# COMMAND ----------

long_term_predictors.write.saveAsTable("dua_058828_spa240.paper1_long_term_care_predictors", mode='overwrite') 