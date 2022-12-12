'''
run queries, verify results, store runtimes
get random collection of spark parameters
because search space is very large and we have deterministic data to cover part of the search space, 
don't worry about collisions with previous runs
@ hoped
'''

from tpch_det import *

def randomize_params():
    new_params = SYSTEM_PARAMETERS[:]
    for param in SPARK_PARAMETERS:
        param = param.copy() # don't mutate spark params
        val_index = np.random.randint(0, len(param['possible_values']))
        param['cur_value'] = param['possible_values'][val_index]
        new_params.append(param)
    return new_params
            
### Examples running script
# python3 tpch_rand.py test 1

if __name__ == "__main__":
    try:
        job_name = sys.argv[1]
        sf = int(sys.argv[2])  # table scale factor
    except:
        job_name = 'alocal_test'
        sf = 1
    N=1 # number of times queries will be run
    print(f"{CURRENT_FILE_PATH=}")
    TABLE_FILE_PATH = f"{CURRENT_FILE_PATH}/../tpch_tables/s{sf}"
   
    # add fixed system parameters
    SYSTEM_PARAMETERS = [make_param('sf', False, 1, [1, 10, 100, 300], sf),
                        make_param('job_name', False, job_name, ['rand', 'det'], job_name)]
    
    for param_name, param_val in getSystemInfo().items():
        SYSTEM_PARAMETERS.append(make_param(param_name, False, None, [], param_val))
    
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_results/sf{sf}_n{N}_rand_{job_name}.txt"
    
    print(f'{LOG_FNAME=}')
    print(f'{sf=}')
    print(f'{TABLE_FILE_PATH=}')
    
    # make random params and run queries
    params = randomize_params()
    result = run_queries(params, TABLE_FILE_PATH)
    print("rand param runtimes", result['runtimes'])
    log_results(result, LOG_FNAME)
    print("done", result['runtimes'])
