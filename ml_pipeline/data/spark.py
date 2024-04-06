#!/usr/bin/env python3
from sys import stdout
import csv

# 'pip install pyspark' for these
from pyspark import SparkFiles
from pyspark.sql import SparkSession

# make a spark "session".  this creates a local hadoop cluster by default (!)
spark = SparkSession.builder.getOrCreate()
# put the input file in the cluster's filesystem:
spark.sparkContext.addFile("https://csvbase.com/meripaterson/stock-exchanges.csv")
# the following is much like for pandas
df = (
    spark.read.csv(f"file://{SparkFiles.get('stock-exchanges.csv')}", header=True)
    .select("MIC")
    .na.drop()
    .sort("MIC")
)
# pyspark has no easy way to write csv to stdout - use python's csv lib
csv.writer(stdout).writerows(df.collect())
