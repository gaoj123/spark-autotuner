from convert_tables import *

# used to verify results have expected table dimensions
# maps scale factor -> qnum -> (# rows, #cols)
QUERY_OUTPUT_DIMENSION_MAP = {
    1: {1: (4, 10), 2: (460, 8), 3: (11620, 4), 4: (5, 2), 5: (5, 2), 6: (1, 1), 7: (4, 4), 8: (2, 2), 9: (175, 3), 10: (37967, 8), 11: (1048, 2), 12: (2, 3),
 13: (42, 2), 14: (1, 1), 15: (1, 5), 16: (18314, 4), 17: (1, 1), 18: (57, 6), 19: (1, 1), 20: (186, 2), 21: (411, 2), 22: (7, 3)},

}

CURRENT_FILE_PATH = os.path.dirname(__file__)
TABLE_FILE_PATH = CURRENT_FILE_PATH + "/../tpch_tables/"
DET_PARAMS_FNAME = CURRENT_FILE_PATH + "/training_params/detparams_9.json"
# currently forces each of the 88 different deterministic parameter sets to run for 10 times so we get a better range of data

DET_PARAMS = None
with open(DET_PARAMS_FNAME, 'rb') as f:
    DET_PARAMS = json.load(f)

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

def run_queries(parameters, debug=False):
    '''
    Run TPC-H queries n times and take the sum of all query runtimes 
    to generate a single training run for a set of parameters.
    
    Input: 
    parameters: list of parameter dictionaries 

    Returns: 
    training_data dictionary with params and runtime results 
    '''
    result = {'params': parameters, 'runtimes': {}}
    spark = None
    # add chosen parameter values to spark
    param_name_index = {}
    print("build spark with params, verify params...")
    start_time = time.time()
    try:
        conf = SparkConf(loadDefaults=False)
        spark_params = []
        for i, param in enumerate(parameters):
            if param['spark_param']:
                # forcing spark shuffle partitions to be low
                # otherwise we get a too many open files error because 
                # ulimit -a is 1024 and we cannot set ulimit to be higher :/
                if param['name'] == "spark.sql.shuffle.partitions":
                     spark_params.append((param['name'], str(100)))
                else:
                    spark_params.append((param['name'], str(param['cur_value'])))
                param_name_index[param['name']] = i
        # for local laptop testing, need to specify local host?
        #spark_params.append(("spark.driver.host", "localhost"))

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
        result = {'params':parameters, 'runtimes':{}, 'msg': str(e)}
        return result
    
    configurations = spark.sparkContext.getConf().getAll()
    # assert configurations have been set properly
    for item in configurations: 
        # ignore shuffle partitions for now because of aforementioned error
        if param_name_index.get(item[0]) is not None and item[0] != "spark.sql.shuffle.partitions":
            assert item[1] == str(parameters[param_name_index.get(item[0])]['cur_value']), f'Spark session param {item} != {parameters[param_name_index.get(item[0])]}'
    end_time = time.time()
    print(end_time-start_time)
    result['runtimes']['build_config'] = end_time-start_time
    
    # load tables
    print("load and verify tables...")
    start_time = time.time()
    tables = {}
    for table_name in TABLE_SCHEMA_MAP:
        table = spark.read.parquet(f"{TABLE_FILE_PATH }/{table_name}.parquet")
        table.createOrReplaceTempView(table_name)
        tables[table_name] = table
        trows, tcols = table.count(), len(table.columns)
        exp_rows, exp_cols = TABLE_DIMENSION_MAP[sf][table_name]
        assert trows == exp_rows, f"sf {sf} table {table_name} got {trows} rows but expected {exp_rows}"
        assert tcols == exp_cols, f"sf {sf} table {table_name} got {tcols} cols but expected {exp_cols}"
        if debug:
            print(f"{table_name=} {trows} rows, {tcols} cols")
    end_time = time.time()
    result['runtimes']['load_tables'] = end_time-start_time
    print(end_time-start_time)
    
    # run queries n times
    print("run queries and verify results...")
    find_expected = sf not in QUERY_OUTPUT_DIMENSION_MAP
    start_time = time.time()
    for qnum, qtext in TPCH_QUERIES.items(): 
        results = spark.sql(qtext, **tables)
        rrows = results.count()
        rcols = len(results.columns)
        if debug:
            print(qnum, rrows, rcols, results)
        if find_expected:
            QUERY_OUTPUT_DIMENSION_MAP.setdefault(sf, {})[qnum] = (rrows, rcols)
        
        exp_rows, exp_cols = QUERY_OUTPUT_DIMENSION_MAP[sf][qnum]
        assert rrows == exp_rows, f"sf {sf} qnum {qnum} got {rrows} rows but expected {exp_rows}"
        assert rcols == exp_cols, f"sf {sf} qnum {qnum} got {rcols} cols but expected {exp_cols}"
        # write results to parquet file?
        # results.write.parquet(f"{TABLE_FILE_PATH}_out/{qnum}.parquet")
    end_time = time.time()
    result['runtimes']['total'] = end_time-start_time
    print(end_time-start_time)
    
    if find_expected:
        print(QUERY_OUTPUT_DIMENSION_MAP)
    
    print(f"total time... {round(sum(result['runtimes'].values()), 2)}") 
    return result
    
def log_results(result_dict, debug=False):
    # log results to a text file by appending a new line containing result dictionary in string form
    with open(LOG_FNAME,'a') as file:
        file.write(str(result_dict)+"\n")

def num_log():
    # return the number of lines currently in the log file (used to determine what set of parameters the script should run)
    try:
        with open(LOG_FNAME,'r') as file:
            # First we load existing data into a dict.
            linecount = len(file.readlines())
            return linecount
    except:
        with open(LOG_FNAME,'w') as file:
            pass
    return 0

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
# commented out, see justification in run_queries func
#make_param('spark.sql.shuffle.partitions', True, 200, list(range(100,1001, 100))),
make_param('spark.default.parallelism', True, 200, list(range(20,400, 40))),
make_param('spark.memory.storageFraction', True, .5, [x/10 for x in list(range(3,9))]), 
]

# Training data stored in log where each entry
'''
{   'params': [ list of param dictionaries as described above], 
    'runtimes': {'build_config': float time(s) representing time to make spark session and set params
                 'load_table': float time (s) representing time to load tables into spark session
                'total': float time(s) it takes to run queries n times }
                
Note runtimes should not be stored in a list-- the first runtime for a query set will be slow, subsequent ones use caching to speed things up
The only reason we have n > 1 is to increase the total runtime in an effort to hopefully reduce noise

For similar reasons we only use a single parameter set per script run
'''
# If a set of parameters led system to crash because they were an impossible combination we will log it like 
# result = {'params':parameters, 'runtimes': {}, 'msg': str(e)}
            

### Examples running script
# python3 tpch_training_deterministic.py test_run_det 1

if __name__ == "__main__":
    try:
        job_name = sys.argv[1]
        sf = int(sys.argv[2])
    except:
        job_name = 'a_hoped_local'
        sf = 1
    TABLE_FILE_PATH = f'{TABLE_FILE_PATH}s{sf}'
    # add fixed system parameters
    SYSTEM_PARAMETERS = [make_param('sf', False, 1, [1, 10, 60, 300], sf), make_param('job_name', False, 1, [1, 10, 60], job_name)]
    for param_name, param_val in getSystemInfo().items():
        SYSTEM_PARAMETERS.append(make_param(param_name, False, None, [], param_val))
   
    # read in 22 TPCH queries
    TPCH_QUERIES = {}
    for i in range(1, 23):
        with open(f"{CURRENT_FILE_PATH}/queries/{i}.sql") as f:
            TPCH_QUERIES[i] = f.read() 
    
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_n1_det_{job_name}.txt"
    SF_STR = f"sf{sf}"
    print(f'{LOG_FNAME=}')
    print(f'{sf=}')
    print(f'{TABLE_FILE_PATH=}')

    
    num_logs = num_log()
    print(f'# {num_logs=}/{len(DET_PARAMS)}')
    if num_logs < len(DET_PARAMS):
        params = DET_PARAMS[str(num_logs)]
        params += SYSTEM_PARAMETERS
        result = run_queries(params)
        print("deterministic runtimes", result['runtimes'])
        log_results(result)
        print(QUERY_OUTPUT_DIMENSION_MAP)
    else:
        print("done")
