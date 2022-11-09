import os 
import sys 
import time 
import numpy as np
import json

from pyspark.sql import SparkSession 
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType)


# Schemas for all table types here. These should be in separate scripts when
# refactoring code.
CUSTOMER_SCHEMA = StructType([
    StructField("c_custkey", LongType()),
    StructField("c_name", StringType()),
    StructField("c_address", StringType()),
    StructField("c_nationkey", LongType()),
    StructField("c_phone", StringType()),
    StructField("c_acctbal", DoubleType()),
    StructField("c_mktsegment", StringType()),
    StructField("c_comment", StringType()),
])

LINEITEM_SCHEMA = StructType([
    StructField("l_orderkey", LongType()),  
    StructField("l_partkey", LongType()),
    StructField("l_suppkey", LongType()),
    StructField("l_linenumber", LongType()),
    StructField("l_quantity", DoubleType()),
    StructField("l_extendedprice", DoubleType()),
    StructField("l_discount", DoubleType()),
    StructField("l_tax", DoubleType()),
    StructField("l_returnflag", StringType()),
    StructField("l_linestatus", StringType()),
    StructField("l_shipdate", StringType()),
    StructField("l_commitdate", StringType()),
    StructField("l_receiptdate", StringType()),
    StructField("l_shipinstruct", StringType()),
    StructField("l_shipmode", StringType()),
    StructField("l_comment", StringType())
])

NATION_SCHEMA = StructType([
    StructField("n_nationkey", LongType()), 
    StructField("n_name", StringType()),
    StructField("n_regionkey", LongType()),
    StructField("n_comment", StringType()),
])

ORDER_SCHEMA = StructType([
    StructField("o_orderkey", LongType()),
    StructField("o_custkey", LongType()),
    StructField("o_orderstatus", StringType()),
    StructField("o_totalprice", DoubleType()),
    StructField("o_orderdate", StringType()),
    StructField("o_orderpriority", StringType()),
    StructField("o_clerk", StringType()),
    StructField("o_shippriority", LongType()),
    StructField("o_comment", StringType())
])

PART_SCHEMA = StructType([
    StructField("p_partkey", LongType()),    
    StructField("p_name", StringType()),
    StructField("p_mfgr", StringType()),
    StructField("p_brand", StringType()),
    StructField("p_type", StringType()),
    StructField("p_size", LongType()),
    StructField("p_container", StringType()),
    StructField("p_retailprice", DoubleType()),
    StructField("p_comment", StringType()),
])

PARTSUPP_SCHEMA = StructType([
    StructField("ps_partkey", LongType()),
    StructField("ps_suppkey", LongType()),
    StructField("ps_availqty", LongType()),
    StructField("ps_supplycost", DoubleType()),
    StructField("ps_comment", StringType())
])

REGION_SCHEMA = StructType([
    StructField("r_regionkey", LongType()),   
    StructField("r_name", StringType()),
    StructField("r_comment", StringType()),  
])

SUPPLIER_SCHEMA = StructType([
    StructField("s_suppkey", LongType()),    
    StructField("s_name", StringType()),
    StructField("s_address", StringType()),
    StructField("s_nationkey", LongType()),
    StructField("s_phone", StringType()),
    StructField("s_acctbal", DoubleType()),
    StructField("s_comment", StringType())
])

CURRENT_FILE_PATH = os.path.dirname(__file__)

if __name__ == "__main__":
    # For now, we expect only a single query to be run at a time for simplicity.
    # Ideally, we should be able to run all queries with a single SparkSession
    # and loading all tables only once.
    try:
        query_num = sys.argv[1]
    except:
        query_num = 1
    
    spark = SparkSession.builder \
                    .appName("TPC- H Benchmarking Queries") \
                    .config("spark.some.config.option", "some-value") \
                    .getOrCreate()
    
    customer = spark.read.csv("./dbgen/customer.tbl", sep = "|", schema=CUSTOMER_SCHEMA)
    customer.createOrReplaceTempView("customer")

    lineitem = spark.read.csv("./dbgen/lineitem.tbl", sep = "|", schema=LINEITEM_SCHEMA)
    lineitem.createOrReplaceTempView("lineitem")

    nation = spark.read.csv("./dbgen/nation.tbl", sep = "|", schema=NATION_SCHEMA)
    nation.createOrReplaceTempView("nation")

    region = spark.read.csv("./dbgen/region.tbl", sep = "|", schema=REGION_SCHEMA)
    region.createOrReplaceTempView("region")

    orders = spark.read.csv("./dbgen/orders.tbl", sep = "|", schema=ORDER_SCHEMA)
    orders.createOrReplaceTempView("orders")

    part = spark.read.csv("./dbgen/part.tbl", sep = "|", schema=PART_SCHEMA)
    part.createOrReplaceTempView("part")

    partsupp = spark.read.csv("./dbgen/partsupp.tbl", sep = "|", schema=PARTSUPP_SCHEMA)
    partsupp.createOrReplaceTempView("partsupp")

    supplier = spark.read.csv("./dbgen/supplier.tbl", sep = "|", schema=SUPPLIER_SCHEMA)
    supplier.createOrReplaceTempView("supplier")
    
    # Read the tables.
    tables = {
        "customer": customer,
        "lineitem": lineitem,
        "nation": nation,
        "region": region,
        "orders": orders,
        "part": part,
        "partsupp": partsupp,
        "supplier": supplier,
    }
   

    result = {}
    # take median of 10 runs for each set of params
    for j in range(10):
        rstart = time.time()
        for i in range(1, 22):
            try: 
                with open("./dbgen/queries/" + str(i) + ".sql") as f:
                    query = f.read() 

                # Measure execution time of sql query.
                start_time = time.time()
                results = spark.sql(query, **tables)
                end_time = time.time()
                result.setdefault(i, []).append(end_time-start_time)
            except:
                print("failed while running query " + str(i) + " ...  ")
            
        rend = time.time()    
        result.setdefault('total', []).append(rend-rstart)
    
    # take median of all runtimes as final
    for key, times in result.items():
        result[key] = np.median(times) 
    
    
    with open('benchmarking.json','r+') as file:
        # First we load existing data into a dict.
        try:
            file_data = json.load(file)
        except: 
            print ("uh can't load filedata")
            file_data = {}
        # Join new_data with file_data
        file_data[str(("hd", time.time()))] = {'params': {'sf': 1}, 'results': result}
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

    with open('benchmarking.json','r') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        
        print(len(file_data))