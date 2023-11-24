# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import numpy as np

# COMMAND ----------

df = spark.table("dua_058828_spa240.enrollment_analysis")
print((df.count(), len(df.columns)))

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("FilterRowsByFirstColumn").getOrCreate()

# Filter the DataFrame to include only rows where the "first" column value corresponds to columns col1 through col24
result_df = df.filter(col("first").isin([f"col{i}" for i in range(1, 25)]))

# Show the filtered DataFrame
print((result_df.count(), len(result_df.columns)))

# COMMAND ----------

df2016 = spark.table("dua_058828_spa240.elig2016")

result_df = result_df.join(df2016, on=["beneID","state"], how="left")
result_df.show()
result_df = result_df.fillna(0)
result_df.show()

# COMMAND ----------

result_df.groupBy('2016months').count().show()

# COMMAND ----------

print((result_df.count(), len(result_df.columns)))  

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from functools import reduce

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("SumColumns").getOrCreate()

# Assuming "result_df" is the DataFrame containing columns "col1" through "col36"

# Create a list of column names from "col1" to "col36"
column_names = [f"col{i}" for i in range(1, 37)]

# Calculate the sum of the values in columns "col1" through "col36" for each row
# and add a new column "totalMonths" with the result
result_df = result_df.withColumn("totalMonths", reduce(lambda a, b: a + b, (col(c) for c in column_names)))

# Show the DataFrame with the new "totalMonths" column
result_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("AssignCoverageYear").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "first" column
# If you have multiple DataFrames (split_1 to split_100), you can concatenate them using the "unionByName" function

# Create a conditional expression to assign the coverage year based on the value of the "first" column
coverage_year_expr = when(col("first").isin([f"col{i}" for i in range(1, 13)]), lit(2017)) \
                     .when(col("first").isin([f"col{i}" for i in range(13, 25)]), lit(2018)) \
                     .otherwise(lit(None))

# Add a new column "coverageYear" to the DataFrame based on the conditional expression
result_df = result_df.withColumn("coverageYear", coverage_year_expr)

# Show the DataFrame with the new "coverageYear" column
result_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, count

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("DistributionOfFirstColumn").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "first" column

# Define the column ranges
range1 = [f"col{i}" for i in range(1, 13)]   # col1 to col12
range2 = [f"col{i}" for i in range(13, 25)]  # col13 to col24

# Create a conditional expression to categorize the values in the "first" column based on the defined ranges
category_expr = when(col("first").isin(range1), "col1-col12") \
                .when(col("first").isin(range2), "col13-col24") \
                .otherwise("Other")

# Add a new column "category" to the DataFrame based on the conditional expression
result_df = result_df.withColumn("category", category_expr)

# Calculate the distribution of the "category" column
distribution_df = result_df.groupBy("category").agg(count("*").alias("count")).orderBy("category")

# Show the distribution of the "category" column
distribution_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("DistributionOfEnrollMonths").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrollMonths" column
# If you have multiple DataFrames (split_1 to split_100), you can concatenate them using the "unionByName" function

# Calculate the total number of rows in the DataFrame
total_rows = result_df.count()

# Calculate the distribution of the "enrollMonths" variable
distribution_df = result_df.groupBy("enrollMonths") \
                    .agg(count("*").alias("count")) \
                    .withColumn("percentage", round((col("count") / total_rows) * 100, 2)) \
                    .orderBy("enrollMonths")  # Order by the "enrollMonths" column

# Show the distribution and percentage of total rows for each month value
distribution_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("DistributionOfTotalMonths").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "totalMonths" column

# Calculate the total number of rows in the DataFrame
total_rows = result_df.count()

# Calculate the distribution of the "totalMonths" variable
distribution_df = result_df.groupBy("totalMonths") \
                    .agg(count("*").alias("count")) \
                    .withColumn("percentage", round((col("count") / total_rows) * 100, 2)) \
                    .orderBy("totalMonths")  # Order the output by "totalMonths" from 0 to 36

# Show the distribution and percentage of total rows for each value of "totalMonths"
distribution_df.show(37)  # Show up to 37 rows to include all values from 0 to 36

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, lit

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("AssignEnrolledStatus").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrollMonths" column
# If you have multiple DataFrames (split_1 to split_100), you can concatenate them using the "unionByName" function

# Create a conditional expression to assign the enrolled status based on the value of the "enrollMonths" column
enrolled_expr = when(col("enrollMonths") == 12, lit(1)).otherwise(lit(0))

# Add a new column "enrolled" to the DataFrame based on the conditional expression
result_df = result_df.withColumn("enrolled", enrolled_expr)

# Show the DataFrame with the new "enrolled" column
result_df.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("PercentageAndCountOfCoverageYear").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "coverageYear" column

# Calculate the total number of rows in the DataFrame
total_rows = result_df.count()

# Calculate the percentage and total count of each value in the "coverageYear" column
coverage_year_distribution = result_df.groupBy("coverageYear") \
    .agg(count("*").alias("count")) \
    .withColumn("percentage", round((col("count") / total_rows) * 100, 2)) \
    .orderBy("coverageYear")  # Order by the "coverageYear" column

# Show the percentage and total count of each value in the "coverageYear" column
coverage_year_distribution.show()

#Count of rows with a value of 1 in columns col1 through col12: 809247
#Count of rows with a value of 1 in columns col13 through col24: 804914

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, sum

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("PercentageOfEnrolled").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrolled" and "coverageYear" columns

# Calculate the total number of rows in the DataFrame
total_rows = result_df.count()

# Calculate the percentage of enrolled individuals for everyone (both 2017 and 2018)
enrolled_percentage_all = result_df.agg(round((sum("enrolled") / total_rows) * 100, 2).alias("enrolled_percentage_all")).collect()[0]["enrolled_percentage_all"]

# Calculate the percentage of enrolled individuals for each year separately (2017 and 2018)
enrolled_percentage_by_year = result_df.groupBy("coverageYear") \
    .agg(round((sum("enrolled") / count("*")) * 100, 2).alias("enrolled_percentage")) \
    .orderBy("coverageYear")

# Show the overall percentage of enrolled individuals
print(f"Overall percentage of enrolled individuals: {enrolled_percentage_all}%")

# Show the percentage of enrolled individuals for each year
enrolled_percentage_by_year.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("EnrollmentByState").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrolled" and "state" columns

# Group the data by the "state" column and calculate the sum of the "enrolled" column (numerator)
# and the total count of members (denominator) for each state
enrollment_by_state = result_df.groupBy("state") \
    .agg(sum("enrolled").alias("enrolled_count"), count("*").alias("total_members"))

# Get the total number of rows in the DataFrame
total_rows = enrollment_by_state.count()

# Show all rows in the DataFrame
enrollment_by_state.show(total_rows)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("MeanTotalMonthsByState").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "totalMonths" and "state" columns

# Group the data by the "state" column and calculate the mean of the "totalMonths" column for each state
mean_total_months_by_state = result_df.groupBy("state") \
    .agg(mean("totalMonths").alias("mean_total_months"))

# Get the total number of rows in the DataFrame
total_rows = mean_total_months_by_state.count()

# Show all rows in the DataFrame
mean_total_months_by_state.show(total_rows)

# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lit, when
# from functools import reduce
# from operator import add

# # Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
# spark = SparkSession.builder.appName("PercentageAfterFirst").getOrCreate()

# # Assuming "result_df" is the DataFrame containing columns "col1" through "col36" and the "first" column

# # Define a function to calculate the percentage of columns with a value of 1 after the first month of enrollment
# def calculate_percentage(df):
#     for i in range(1, 37):
#         # Create a list of columns to sum after the first month of enrollment
#         cols_to_sum = [col(f"col{j}") for j in range(i + 1, 37)]
#         # Calculate the sum of the columns with value 1
#         if cols_to_sum:
#             sum_expr = reduce(add, [when(c == 1, 1).otherwise(0) for c in cols_to_sum])
#         else:
#             sum_expr = lit(0)
#         # Calculate the percentage for each row based on the "first" column value
#         df = df.withColumn("percentage_after_first", 
#                            when(col("first") == f"col{i}", sum_expr / (37 - (i + 1)))
#                            .otherwise(col("percentage_after_first")))
#     return df

# # Process the DataFrame
# result_df = calculate_percentage(result_df)

# # Show the DataFrame with the new "percentage_after_first" column
# result_df.show()

# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import avg

# # Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
# spark = SparkSession.builder.appName("AveragePercentageAfterFirstByState").getOrCreate()

# # Assuming "result_df" is the DataFrame containing the "state" and "percentage_after_first" columns

# # Group the data by the "state" column and calculate the average of the "percentage_after_first" column for each group
# average_percentage_by_state = result_df.groupBy("state") \
#     .agg(avg("percentage_after_first").alias("average_percentage_after_first"))

# # Get the total number of rows in the DataFrame
# total_rows = average_percentage_by_state.count()

# # Show all rows of the average percentage after the first month of enrollment for each state
# average_percentage_by_state.show(total_rows)

# COMMAND ----------

result_df.write.saveAsTable("dua_058828_spa240.stage1_analysis", mode="overwrite")

# COMMAND ----------

df2017 = spark.table("dua_058828_spa240.finalSample2017")
df2018 = spark.table("dua_058828_spa240.finalSample2018")
df2019 = spark.table("dua_058828_spa240.finalSample2019")

# COMMAND ----------

print(df2017.printSchema())

# COMMAND ----------

member2017 = df2017.select("beneID", "state", "county", "ageCat","age","sex","race","speakEnglish","married","houseSize","fedPovLine","UsCitizen","ssdi","ssi","tanf","disabled")
member2018 = df2018.select("beneID", "state", "county", "ageCat","age","sex","race","speakEnglish","married","houseSize","fedPovLine","UsCitizen","ssdi","ssi","tanf","disabled")
member2019 = df2019.select("beneID", "state", "county", "ageCat","age","sex","race","speakEnglish","married","houseSize","fedPovLine","UsCitizen","ssdi","ssi","tanf","disabled")
memberAll = member2017.union(member2018).union(member2019)
print(memberAll.printSchema())

# COMMAND ----------

memberAll.registerTempTable("connections")

memberAgg = spark.sql('''
SELECT beneID, state, max(county) as county, max(age) as age, max(ageCat) as ageCat, max(sex) as sex, max(race) as race, max(speakEnglish) as speakEnglish, max(married) as married, max(houseSize) as houseSize, max(fedPovLine) as fedPovLine, max(UsCitizen) as UsCitizen, max(ssdi) as ssdi, max(ssi) as ssi, max(tanf) as tanf, max(disabled) as disabled 
FROM connections
GROUP BY beneID, state;
''')

# COMMAND ----------

print((memberAll.count(), len(memberAll.columns)))
print((memberAgg.count(), len(memberAgg.columns)))

# COMMAND ----------

print(result_df.printSchema())
result = result_df.select("beneID", "state", "first", "enrolled","enrollMonths","2016months")
print(result.printSchema())

# COMMAND ----------

from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Left Join DataFrames") \
    .getOrCreate()

# Remove duplicates based on join columns from both DataFrames
#df1 = df1.dropDuplicates(["beneID", "state"])
#df2 = df2.dropDuplicates(["beneID", "state"])

# Perform left join on "beneID" and "state" columns
step1_final = result.join(memberAgg, on=["beneID", "state"], how="left")

# Show the result of the join
step1_final.show()

# COMMAND ----------

print((step1_final.count(), len(step1_final.columns)))
print((result.count(), len(result.columns)))
print((memberAgg.count(), len(memberAgg.columns)))

# COMMAND ----------

step1_final.write.saveAsTable("dua_058828_spa240.stage1_final_analysis", mode="overwrite")

# COMMAND ----------

step1_final = spark.table("dua_058828_spa240.stage1_final_analysis")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("EnrollmentByState").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrolled" and "state" columns

# Group the data by the "state" column and calculate the sum of the "enrolled" column (numerator)
# and the total count of members (denominator) for each state
enrollment_by_state = step1_final.groupBy("state") \
    .agg(sum("enrolled").alias("enrolled_count"), count("*").alias("total_members"))

# Get the total number of rows in the DataFrame
total_rows = enrollment_by_state.count()

# Show all rows in the DataFrame
enrollment_by_state.show(total_rows)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count

# Create a Spark session (this is not necessary in Databricks, as the Spark session is already created for you)
spark = SparkSession.builder.appName("EnrollmentByState").getOrCreate()

# Assuming "result_df" is the DataFrame containing the "enrolled" and "state" columns

# Group the data by the "state" column and calculate the sum of the "enrolled" column (numerator)
# and the total count of members (denominator) for each state
enrollment_by_race = step1_final.groupBy("race") \
    .agg(sum("enrolled").alias("enrolled_count"), count("*").alias("total_members"))

# Get the total number of rows in the DataFrame
total_rows = enrollment_by_race.count()

# Show all rows in the DataFrame
enrollment_by_race.show(total_rows)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count

# Group the data by the "state" column and calculate the sum of the "enrolled" column (numerator)
# and the total count of members (denominator) for each state
enrollment_by_age = step1_final.groupBy("ageCat") \
    .agg(sum("enrolled").alias("enrolled_count"), count("*").alias("total_members"))

# Get the total number of rows in the DataFrame
total_rows = enrollment_by_age.count()

# Show all rows in the DataFrame
enrollment_by_age.show(total_rows)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, count

# Group the data by the "state" column and calculate the sum of the "enrolled" column (numerator)
# and the total count of members (denominator) for each state
enrollment_by_sex = step1_final.groupBy("sex") \
    .agg(sum("enrolled").alias("enrolled_count"), count("*").alias("total_members"))

# Get the total number of rows in the DataFrame
total_rows = enrollment_by_sex.count()

# Show all rows in the DataFrame
enrollment_by_sex.show(total_rows)