'''
run queries against a deterministic set of params meant to compare the results of params
built from different models / hand tuning

collect 20 samples from each param set in round robin order, find which has best median / avg time
'''

from tpch_det import *


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
# python3 tpch_compare.py test 1
        
if __name__ == "__main__":
    try:
        job_name = sys.argv[1]
        sf = int(sys.argv[2])  # table scale factor
    except:
        job_name = 'alocal_test'
        sf = 1
    N = 20
    print(f"{CURRENT_FILE_PATH=}")
    TABLE_FILE_PATH = f"{CURRENT_FILE_PATH}/../tpch_tables/s{sf}"
    DET_PARAMS_FNAME = f"{CURRENT_FILE_PATH}/training_params/compareparams_n{N}.json"
    DET_PARAMS = None
    with open(DET_PARAMS_FNAME, 'rb') as f:
        DET_PARAMS = json.load(f)

    # add fixed system parameters
    SYSTEM_PARAMETERS = [make_param('sf', False, 1, [1, 10, 100, 300], sf),
                        make_param('job_name', False, job_name, ['rand', 'det'], job_name)]
    
    for param_name, param_val in getSystemInfo().items():
        SYSTEM_PARAMETERS.append(make_param(param_name, False, None, [], param_val))
   
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_n{N}_compare_{job_name}.txt"
    
    print(f'{LOG_FNAME=}')
    print(f'{sf=}')
    print(f'{TABLE_FILE_PATH=}')

    # run comparison parameters
    num_logs = num_log()
    print(f'running param set # {num_logs=}/{len(DET_PARAMS)}')
    if num_logs < len(DET_PARAMS):
        params = DET_PARAMS[str(num_logs)]
        params += SYSTEM_PARAMETERS
        result = run_queries(params, TABLE_FILE_PATH)
        print("compare runtimes", result['runtimes'])
        log_results(result, LOG_FNAME)
    else:
        print("done")
