'''
run queries, verify results, store runtimes
deterministically vary parameters one at a time to all possible values
useful for sensitivity analysis, hand tuning
@ hoped
'''

# import table schemas spark stuff
from convert_tables import *
# import spark params / deterministic param sets / random param generator funcs
from tpch_param import * 

# used to verify results have expected table dimensions
# maps qnum -> (# rows, #cols) for scale factor 1
QUERY_OUTPUT_DIMENSION_MAP = { 1: (4, 10), 2: (460, 8), 3: (11620, 4), 4: (5, 2), 5: (5, 2), 6: (1, 1), 7: (4, 4), 8: (2, 2),
         9: (175, 3), 10: (37967, 8), 11: (1048, 2), 12: (2, 3), 13: (42, 2), 14: (1, 1), 15: (1, 5), 
         16: (18314, 4), 17: (1, 1), 18: (57, 6), 19: (1, 1), 20: (186, 2), 21: (411, 2), 22: (7, 3),
}

CURRENT_FILE_PATH = os.path.dirname(__file__)
# read in 22 TPCH queries
TPCH_QUERIES = {}
for i in range(1, 23):
    with open(f"{CURRENT_FILE_PATH}/queries/{i}.sql") as f:
        TPCH_QUERIES[i] = f.read() 

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

def run_queries(parameters, table_path):
    '''
    Run TPC-H queries n times and take the sum of all query runtimes 
    to generate a single training run for a set of parameters.
    
    Input: 
    parameters: list of parameter dictionaries 
    table_path: string containing file path to TPCH tables

    Returns: 
    training_data dictionary with params and runtime results 
    '''
    result = {'params': parameters, 'runtimes': {}}
    spark = None
   
    try:  # giant try except: great style
        # add chosen parameter values to spark
        param_name_index = {}
        print("~~~       build spark with params, verify params...")
        start_time = time.time()
        conf = SparkConf(loadDefaults=False)
        spark_params = []
        for i, param in enumerate(parameters):
            if param['spark_param']:
                spark_params.append((param['name'], str(param['cur_value'])))
                param_name_index[param['name']] = i
        # for local laptop testing, need to specify local host?
        #spark_params.append(("spark.driver.host", "localhost"))

        conf.setAll(spark_params)
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        spark.sparkContext.setLogLevel("OFF") # INFO, WARN
        # assert configurations have been set properly
        # disabled to allow for faster training data collection
        '''
        configurations = spark.sparkContext.getConf().getAll()
        for item in configurations:
            if param_name_index.get(item[0]):
                assert item[1] == str(parameters[param_name_index.get(item[0])]['cur_value']), \
                    f'Spark session param {item} != {parameters[param_name_index.get(item[0])]}'
        '''
        end_time = time.time()
        print(f"          {end_time-start_time} seconds")
        result['runtimes']['build_config'] = end_time-start_time

        # load tables
        print("~~~       load and verify tables...")
        start_time = time.time()
        tables = {}
        for table_name in TABLE_SCHEMA_MAP:
            # load each table
            table = spark.read.parquet(f"{table_path}/{table_name}.parquet")
            
            # running with parquet files instead of .tbl
            # huge runtime difference - 488s for .tbl vs 170s for .parquet
            #table = spark.read.csv(f"{TABLE_FILE_PATH}/{table_name}.tbl", sep = "|",
            #                   schema=TABLE_SCHEMA_MAP[table_name])
            table.createOrReplaceTempView(table_name)
            tables[table_name] = table
            # find num rows and columns, verify they match expected values
            # disabled to allow for faster training data collection
            '''
            trows, tcols = table.count(), len(table.columns)
            exp_rows, exp_cols = TABLE_DIMENSION_MAP[sf][table_name]
            assert trows == exp_rows, f"sf {sf} table {table_name} got {trows} rows but expected {exp_rows}"
            assert tcols == exp_cols, f"sf {sf} table {table_name} got {tcols} cols but expected {exp_cols}"
            '''
        end_time = time.time()
        result['runtimes']['load_tables'] = end_time-start_time
        print(f"          {end_time-start_time} seconds")

        print("~~~       run queries and verify results...")
        for qnum, qtext in TPCH_QUERIES.items():
            # run query, measure time
            start_time = time.time()
            results = spark.sql(qtext, **tables)
            rrows = results.count()
            rcols = len(results.columns)
            end_time = time.time()
            result['runtimes'][f'q{qnum}'] = end_time-start_time
            # verify results -- will only work for scale factor 1 !
            '''
            exp_rows, exp_cols = QUERY_OUTPUT_DIMENSION_MAP[qnum]
            assert rrows == exp_rows, f"qnum {qnum} got {rrows} rows but expected {exp_rows}"
            assert rcols == exp_cols, f"qnum {qnum} got {rcols} cols but expected {exp_cols}"
            '''
        # total runtime for all queries
        result['runtimes']['total'] = sum(result['runtimes'][k] for k in result['runtimes'] if 'q' in k)
        print(f"          {end_time-start_time} seconds")
        # overall runtime for all steps
        result['runtimes']['overall'] = sum(result['runtimes'][k] for k in result['runtimes'] if 'q' not in k)

        print(f"~~~       total time... {round(result['runtimes']['total'], 2)}")
    except Exception as e:
        # this might happen because some parameters are related,
        # and we might have made an impossible parameter assignment
        result = {'params':parameters, 'runtimes':{}, 'msg': str(e)}
        print(f"~~~       LOGGING JOB FAILURE   ~~~")
        print(e)

    return result
    
def log_results(result_dict, log_fname): 
    '''
    log results to a text file by appending a new line containing result dictionary in string form.
    Each result dict is of the form
    {   'params': [ list of param dictionaries as described in tpch_param.py ], 
        'runtimes': {'build_config': float time(s) representing time to make spark session and set params
                     'load_table': float time (s) representing time to load tables into spark session
                     'q1': float time (s) time it takes to run query 1,
                     ...
                     'q22': float time (s) it tames to run query 22,
                     'total': float time(s) it takes to run and verify all 22 queries
                     'overall': float time(s) combining all the above}
    
    If a set of parameters led system to crash we will log it like 
     result = {'params':parameters, 'runtimes': {}, 'msg': str(e)}
    '''
    
    with open(log_fname,'a') as file:
        file.write(str(result_dict)+"\n")

def num_log():
    # return the number of lines currently in the log file 
    # (used to determine which set of parameters the script should run)
    try:
        with open(LOG_FNAME,'r') as file:
            # First we load existing data into a dict.
            linecount = len(file.readlines())
            return linecount
    except:
        try:
            with open(LOG_FNAME,'w') as file:
                pass
        except:
            LOG_FNAME 
    return 0
        

### Examples running script
# python3 tpch_det.py test 1 10

if __name__ == "__main__":
    try:
        job_name = sys.argv[1]
        sf = int(sys.argv[2])  # table scale factor
        N = int(sys.argv[3])   # number of times deterministic parameter runtimes are measured
    except:
        job_name = 'alocal_test'
        sf = 1
        N = 10
        
    print(f"{CURRENT_FILE_PATH=}")
    TABLE_FILE_PATH = f"{CURRENT_FILE_PATH}/../tpch_tables/s{sf}"
    DET_PARAMS_FNAME = f"{CURRENT_FILE_PATH}/training_params/detparams_n{N}.json"
    DET_PARAMS = None
    with open(DET_PARAMS_FNAME, 'rb') as f:
        DET_PARAMS = json.load(f)

    # add fixed system parameters
    SYSTEM_PARAMETERS = [make_param('sf', False, 1, [1, 10, 100, 300], sf),
                        make_param('job_name', False, job_name, ['rand', 'det'], job_name)]
    
    for param_name, param_val in getSystemInfo().items():
        SYSTEM_PARAMETERS.append(make_param(param_name, False, None, [], param_val))
   
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_n{N}_det_{job_name}.txt"
    
    print(f'{LOG_FNAME=}')
    print(f'{sf=}')
    print(f'{TABLE_FILE_PATH=}')

    # run deterministic parameters
    num_logs = num_log()
    print(f'running param set # {num_logs=}/{len(DET_PARAMS)}')
    if num_logs < len(DET_PARAMS):
        params = DET_PARAMS[str(num_logs)]
        params += SYSTEM_PARAMETERS
        result = run_queries(params, TABLE_FILE_PATH)
        print("deterministic runtimes", result['runtimes'])
        log_results(result, LOG_FNAME)
    else:
        print("done")
