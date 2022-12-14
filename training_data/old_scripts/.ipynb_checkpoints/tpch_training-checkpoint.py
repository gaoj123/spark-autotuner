import os 
import sys 
import time 
import numpy as np
import json
import gc

from pyspark.sql import SparkSession 
from pyspark.conf import SparkConf

from pyspark.context import SparkContext
from pyspark.sql.types import (
    DoubleType, LongType, StringType, StructField, StructType)

import platform,socket,re,uuid,json,psutil,logging


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

TABLE_SCHEMA_MAP = {
        "customer": CUSTOMER_SCHEMA,
        "lineitem": LINEITEM_SCHEMA,
        "nation": NATION_SCHEMA,
        "region": REGION_SCHEMA,
        "orders": ORDER_SCHEMA,
        "part": PART_SCHEMA,
        "partsupp": PARTSUPP_SCHEMA,
        "supplier": SUPPLIER_SCHEMA,
}

CURRENT_FILE_PATH = os.path.dirname(__file__)
TABLE_FILE_PATH = CURRENT_FILE_PATH + "/../TPC-H V3.0.1/dbgen"

def get_scale_factor():
    # look at table files sizes and add them together
    # return scale factor of all table sizes combined as an int representing GB
    files = os.listdir(TABLE_FILE_PATH)
    files = [f for f in files if os.path.isfile(f'{TABLE_FILE_PATH}/{f}') and '.tbl' in f]
    table_sizes = [os.path.getsize(f'{TABLE_FILE_PATH}/{f}') for f in files]
    return round(sum(table_sizes)/ (1024.0 **3))

def getSystemInfo():
    info={}
    try:
        info['platform']=platform.system()
        info['platform-release']=platform.release()
        info['platform-version']=platform.version()
        info['architecture']=platform.machine()
        info['num_cpus'] = os.cpu_count()
        info['hostname']=socket.gethostname()
        info['ip-address']=socket.gethostbyname(socket.gethostname())
        info['mac-address']=':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor']=platform.processor()
        info['ram']=str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
        info['total_storage']=str(round(psutil.disk_usage(CURRENT_FILE_PATH).total / (1024.0 **3)))+" GB"
        info['free_storage']=str(round(psutil.disk_usage(CURRENT_FILE_PATH).free / (1024.0 **3)))+" GB"
    except Exception as e:
        logging.exception(e)
    return info

def run_queries(parameters, n=10, find_median_runtime=True):
    '''
    Run TPC-H queries 10 times and take the median runtime of each query 
    to generate a single training run for a set of parameters.
    
    Input: 
    parameters: list of parameter dictionaries 
    n: int number of times to run queries
    find_median_runtime: if True returns runtimes values as a float (median), otherwise as a list of all n runtimes 
    
    Returns: 
    training_data dictionary with params and results 
    '''
    result = {'params': [p.copy() for p in parameters], 'runtimes': {'total': []}}
    spark = None
    # add chosen parameter values to spark
    param_name_index = {}
    try:
        conf = SparkConf(loadDefaults=False)
        spark_params = []
        for i, param in enumerate(parameters):
            if param['spark_param']:
                spark_params.append((param['name'], str(param['cur_value'])))
                param_name_index[param['name']] = i

        conf.setAll(spark_params)
        spark  = SparkSession.builder.config(conf=conf).getOrCreate()
        spark.catalog.clearCache() # clear cache
                
    except Exception as e:
        if spark:
            spark.stop()
            del spark
            gc.collect()
        # this might happen because some parameters are related,
        # and we might have made an impossible parameter assignment
        result = {'params':parameters, 'runtimes': {}, 'msg': str(e)}
        return result
    
    configurations = spark.sparkContext.getConf().getAll()
    # assert configurations have been set properly
    for item in configurations: 
        if param_name_index.get(item[0]) is not None:
            assert item[1] == str(parameters[param_name_index.get(item[0])]['cur_value']), f'Spark session param {item} != {parameters[param_name_index.get(item[0])]}'
    
    # load tables
    start_time = time.time()
    tables = {}
    for table_name, table_schema in TABLE_SCHEMA_MAP.items():
        table = spark.read.csv(f"{TABLE_FILE_PATH}/{table_name}.tbl", sep = "|",
                               schema=table_schema)
        table.createOrReplaceTempView(table_name)
        tables[table_name] = table
    end_time = time.time()
    result['runtimes']['load_tables'] = [end_time-start_time]
    # take median of n runs for each query
    for j in range(n):
        spark.catalog.clearCache() # clear cache before each run
        result['runtimes']['total'].append(0)
        for qnum, qtext in TPCH_QUERIES.items(): 
            # Measure execution time of sql query.
            start_time = time.time()
            results = spark.sql(qtext, **tables)
            end_time = time.time()
            query_time = end_time - start_time
            result['runtimes'].setdefault(qnum, []).append(query_time)
            result['runtimes']['total'][-1] += query_time

    if find_median_runtime:
        # take median of all runtimes as final output
        for key, times in result['runtimes'].items():
            result['runtimes'][key] = np.median(times)

    # reset spark so we can load new param config next time
    spark.catalog.clearCache() # clear cache at the end of each run just in case?
    spark.stop()
    del spark
    del tables
    gc.collect()

    return result
    
def log_results(result_dict, debug=False):
    # store query run results in a json file
    # TODO -- there has to be a more efficient way of logging than this...
    with open(LOG_FNAME,'r+') as file:
        # First we load existing data into a dict.
        try:
            file_data = json.load(file)
        except: 
            if debug:
                print ("uh can't load filedata")
            file_data = {}
        # Join new_data with file_data
        file_data[str(("hd", time.time(), len(file_data)+1))] = result_dict
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

# Represent possible spark configuration parameters
def make_param(name, spark_param, default_val, possible_vals, cur_val=None):
    return { 'name': name, 
             'spark_param': spark_param, 
             'cur_value': cur_val if cur_val is not None else default_val, 
             'default_value': default_val, 
             'possible_values': possible_vals
           }
'''
List of param dictionaries, where each dictionary represents a single parameter
{
'spark_param': True / False -- system parameters are fixed, not something we feed to spark but something we should record
'name': str
'cur_value': # initially set to default - this is what we will be changing / generating on each run
'default_value': num - fixed
'possible_values': list of possible values
}
'''
SPARK_PARAMETERS = [make_param('spark.executor.cores', True, 1, list(range(1,9))),
make_param('spark.executor.memory', True, '1g',[f'{x}g' for x in range(1,9)]),
make_param('spark.executor.instances', True, 2, list(range(2,9))),
make_param('spark.driver.cores', True, 1, list(range(1,5))),
make_param('spark.driver.memory', True, '1g',[f'{x}g' for x in range(1,5)]),
make_param('spark.reducer.maxSizeInFlight', True, '48m', [f'{x}m' for x in list(range(48,100,8))]), 
make_param('spark.shuffle.compress', True, 'true', ['true', 'false']),
make_param('spark.shuffle.spill.compress', True, 'true', ['true', 'false']),
make_param('spark.shuffle.file.buffer', True, '32k', [f'{x}k' for x in list(range(32,132,16))]), 
make_param('spark.broadcast.blockSize', True, '4m', [f'{x}m' for x in list(range(4,25,2))]), 
make_param('spark.broadcast.compress', True, 'true', ['true', 'false']),
make_param('spark.memory.fraction', True, .6, [x/10 for x in list(range(3,9))]), 
make_param('spark.rpc.message.maxSize', True, '128', [f"{x}" for x in list(range(128,260, 32))]), 
make_param('spark.rdd.compress', True, 'false', ['true', 'false']),
make_param('spark.io.compression.codec', True, 'lz4', ['lz4', 'snappy']),
make_param('spark.task.cpus', True, 1, list(range(1,3))),
make_param('spark.sql.shuffle.partitions', True, 100, list(range(50, 141, 10))), 
make_param('spark.default.parallelism', True, 200, list(range(20,400, 40))),
make_param('spark.memory.storageFraction', True, .5, [x/10 for x in list(range(3,9))]), 
]
'''
Note:
forcing spark shuffle.partitions to be low (normally default is 200, range from 100-1000 in increments of 100)
because otherwise we get a too many open files error 
ulimit -a is 1024 and we cannot set ulimit -n to be higher  on engaging platform :/
'''

# Training data stored in json where each entry
'''
{   'params': [ list of param dictionaries as described above], 
    'runtimes': {1: float time (seconds), median runtime of query1
                ....
                 22: 
                'total': float time(s)) }
'''
# If a set of parameters led system to crash because they were an impossible combination we will log it like 
# result = {'params':parameters, 'runtimes': {}, 'msg': str(e)}

def deterministic_param_runs(find_median_runtime=True):
    '''
    deterministic approach:
    loop through all possible parameters, and all possible values for each parameter. 
    Change only one parameter at a time, keeping the rest of the params default and
    measure runtime so we can have a baseline for how each individual param impacts performance
    before we start running different combinations of params.
    Good for sensitivity analysis hopefully
    '''
    params = SPARK_PARAMETERS + SYSTEM_PARAMETERS
    # run with default params first
    result = run_queries(params, 100, find_median_runtime)
    log_results(result)
    
    for param in params:
        if param['spark_param']:
            for val in param['possible_values']:
                if val != param['default_value']:
                    param['cur_value'] = val
                    result = run_queries(params, 100, find_median_runtime)               
                    log_results(result)

            # reset param back to default
            param['cur_value'] = param['default_value']
    

                    
def randomize_params():
    new_params = SYSTEM_PARAMETERS[:]
    for param in SPARK_PARAMETERS:
        param = param.copy() # don't mutate spark params
        val_index = np.random.randint(0, len(param['possible_values']))
        param['cur_value'] = param['possible_values'][val_index]
        new_params.append(param)
    return new_params
            

### Examples running script
# python3 tpch_training.py test_run
                            
# Example Slurm job
'''
#!/bin/bash 
#SBATCH -n 4 #Request 4 tasks (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH -t 0-06:00 #Request runtime of 1 minutes
#SBATCH -C centos7 #Request only Centos7 nodes
#SBATCH -p sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=4000 #Request 4G of memory per CPU
#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt
echo $PATH
echo "hi"
module add python/3.9.4
alias python='/usr/bin/python3.9.4'
python3 ??? version 
pip3 install --user numpy
pip3 install --user pyspark

python3  ../../../../../../../spark-autotuner/training_data/tpch_training.py main_job_n4_mem-per-cpu4000

'''

if __name__ == "__main__":
    try:
        job_name = sys.argv[1]
    except:
        job_name = 'local_run'
    sf = get_scale_factor()
    # add fixed system parameters
    SYSTEM_PARAMETERS = [make_param('sf', False, 1, [1, 10, 60, 300], sf), make_param('job_name', False, 1, [1, 10, 60], job_name)]
    for param_name, param_val in getSystemInfo().items():
        SYSTEM_PARAMETERS.append(make_param(param_name, False, None, [], param_val))
        
    # get number of files 
    direc = f"{CURRENT_FILE_PATH}/training_results/"
    files = os.listdir(direc)
    files = [f for f in files if os.path.isfile(direc+'/'+f)] #just files
    num_files = len(files)
    
    # read in 22 TPCH queries
    TPCH_QUERIES = {}
    for i in range(1, 23):
        with open(f"{CURRENT_FILE_PATH}/queries/{i}.sql") as f:
            TPCH_QUERIES[i] = f.read() 
    
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_{job_name}_deterministic_raw_runtimes_{num_files+1}.json"
    SF_STR = f"sf{sf}"
    print(LOG_FNAME)
    
    with open(LOG_FNAME, "wb") as f:
        pass # create empty file? 

    print("starting deterministic list of runtime runs")
    deterministic_param_runs(False)
    
    # make a unique log file name for medians
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_{job_name}_deterministic_and_random_{num_files+1}.json"
    SF_STR = f"sf{sf}"
    print(LOG_FNAME)
    
    with open(LOG_FNAME, "wb") as f:
        pass # create empty file? 

    print("starting deterministic median runs")
    deterministic_param_runs()
    
    print("moving to randomize params")

    while True:
        # generate next set of parameters by randomizing every spark parameter
        params = randomize_params()
        result = run_queries(params, 100)               
        log_results(result)

