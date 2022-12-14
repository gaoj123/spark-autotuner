'''
deterministically vary spark parameters one at a time to all possible values,
store possible values in a json file that will be read by the deterministic script

@ hoped
'''

import os 
import json
import sys

CURRENT_FILE_PATH = os.path.dirname(__file__)
def log_results(result_dict, debug=False):
    # store query run results in a json file
    # TODO -- there has to be a more efficient way of logging than this...
    with open(LOG_FNAME,'r+') as file:
        # First we load existing data into a dict.
        try:
            file_data = json.load(file)
            count = len(file_data)
        except: 
            if debug:
                print ("uh can't load filedata")
            file_data = {}
            count = 0
        # Join new_data with file_data
        file_data[str(count)] = result_dict
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
    params = SPARK_PARAMETERS
    # run with default params first
    for i in range(N):
        log_results(params)
    
        for param in params:
            if param['spark_param']:
                for val in param['possible_values']:
                    if val != param['default_value']:
                        param['cur_value'] = val
                        log_results(params)

                # reset param back to default
                param['cur_value'] = param['default_value']
        print(f"done logging {i=}")


def comparison_param_runs(param_lists):
    # round robin repeat each param list N times
    for i in range(N): 
        for pl in param_lists:
            log_results(pl)

def get_param_lists():
    # default spark params
    deft = {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '2', 'spark.driver.cores': '1', 'spark.driver.memory': '1g', 'spark.reducer.maxSizeInFlight': '48m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '32k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '128', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'lz4', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '100', 'spark.default.parallelism': '200', 'spark.memory.storageFraction': '0.5'}

    # parameters and values chosen as a result of hand tuning
    hand_tuning = {
    'spark.broadcast.blockSize':  '14m',
    'spark.broadcast.compress':  'false',
    'spark.default.parallelism':  '20',
    'spark.driver.cores':  '3',
    'spark.driver.memory':  '4g',
    'spark.executor.cores':  '8',
    'spark.executor.instances':  '8',
    'spark.executor.memory':  '4g',
    'spark.io.compression.codec':  'snappy',
    'spark.memory.fraction':  '0.8',
    'spark.memory.storageFraction':  '0.7',
    'spark.rdd.compress':  'true', 
    'spark.reducer.maxSizeInFlight':  '80m',
    'spark.rpc.message.maxSize': '192',
    'spark.shuffle.compress':  'false',
    'spark.shuffle.file.buffer':  '48k',
    'spark.shuffle.spill.compress':  'false', 
    'spark.sql.shuffle.partitions':  '50',
    'spark.task.cpus':  '1',  
    }
    
    hand_tuning10 = {
    'spark.broadcast.blockSize':  '24m',
    'spark.broadcast.compress':  'false',
    'spark.default.parallelism':  '60',
    'spark.driver.cores':  '3',
    'spark.driver.memory':  '4g',
    'spark.executor.cores':  '3',
    'spark.executor.instances':  '8',
    'spark.executor.memory':  '7g',
    'spark.io.compression.codec':  'snappy',
    'spark.memory.fraction':  '0.5',
    'spark.memory.storageFraction':  '0.6',
    'spark.rdd.compress':  'true', 
    'spark.reducer.maxSizeInFlight':  '72m',
    'spark.rpc.message.maxSize': '224',
    'spark.shuffle.compress':  'false',
    'spark.shuffle.file.buffer':  '32k',
    'spark.shuffle.spill.compress':  'false', 
    'spark.sql.shuffle.partitions':  '80',
    'spark.task.cpus':  '1',  
    }
    
    # Random best from all training data
    rand_tuning = {'spark.executor.cores': 3,
 'spark.executor.memory': '1g',
 'spark.executor.instances': 7,
 'spark.driver.cores': 3,
 'spark.driver.memory': '4g',
 'spark.reducer.maxSizeInFlight': '64m',
 'spark.shuffle.compress': 'true',
 'spark.shuffle.spill.compress': 'true',
 'spark.shuffle.file.buffer': '96k',
 'spark.broadcast.blockSize': '6m',
 'spark.broadcast.compress': 'false',
 'spark.memory.fraction': 0.6,
 'spark.rpc.message.maxSize': '192',
 'spark.rdd.compress': 'true',
 'spark.io.compression.codec': 'snappy',
 'spark.task.cpus': 1,
 'spark.sql.shuffle.partitions': 60,
 'spark.default.parallelism': 60,
 'spark.memory.storageFraction': 0.4}
    
    # MDP RESULTS
    MDP_RESULTS = [
    #0-cost-metric-133.8437633514404 = 
        {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '56m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '128', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '100', 'spark.memory.storageFraction': '0.3' },
    #1-cost-metric-133.8437633514404 = 
        {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '48m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.8', 'spark.rpc.message.maxSize': '224', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.4' },
    #2-cost-metric-133.8437633514404 = 
        {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '8', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '56m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '50', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.3' },
    #3-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '6', 'spark.driver.cores': '2', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '48m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '8m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '100', 'spark.memory.storageFraction': '0.4' },
    #4-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '4', 'spark.executor.memory': '1g', 'spark.executor.instances': '8', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '64m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.7', 'spark.rpc.message.maxSize': '128', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '50', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.4' },
    #5-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '4', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '64k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.7', 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '60', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.4' },
    ]
    '''
        # Other MDP Param sets not included
    6-cost-metric-133.84376335144043 = {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '64m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '128k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '224', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '60', 'spark.default.parallelism': '100', 'spark.memory.storageFraction': '0.4' },
    7-cost-metric-133.84376335144043 = {'spark.executor.cores': '5', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '56m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.5' },
    8-cost-metric-133.84376335144043 = {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '8', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '64m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '8m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '60', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.5' },
    #9-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '2', 'spark.executor.memory': '1g', 'spark.executor.instances': '6', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '48m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '20', 'spark.memory.storageFraction': '0.5' },
    #10-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '56m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '224', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.4' },
    #11-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '4', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '72m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '50', 'spark.default.parallelism': '20', 'spark.memory.storageFraction': '0.4' },
    #12-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '8', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '64m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '64k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.8', 'spark.rpc.message.maxSize': '128', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.3' },
    #13-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '8', 'spark.driver.cores': '2', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '80k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '100', 'spark.memory.storageFraction': '0.3' },
    #14-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '1', 'spark.executor.memory': '1g', 'spark.executor.instances': '6', 'spark.driver.cores': '2', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '48m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '128k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.4', 'spark.rpc.message.maxSize': '224', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '20', 'spark.memory.storageFraction': '0.3' },
    #15-cost-metric-133.84376335144043 = 
        {'spark.executor.cores': '2', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.5', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '60', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.3' }
        '''
    
    '''
    # OLD MDP_RESULTS
    MDP_RESULTS = [ {'spark.executor.cores': '3', 'spark.executor.memory': '1g', 'spark.executor.instances': '7', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '64m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '60', 'spark.default.parallelism': '60', 'spark.memory.storageFraction': '0.4' }, 
    #cost-metric-167.30470418930054 ^
    {'spark.executor.cores': '7', 'spark.executor.memory': '7g', 'spark.executor.instances': '5', 'spark.driver.cores': '3', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '32k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.7', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '80', 'spark.default.parallelism': '260', 'spark.memory.storageFraction': '0.6' }, 
    # cost-metric-169.96870923042297 ^
    {'spark.executor.cores': '1', 'spark.executor.memory': '4g', 'spark.executor.instances': '2', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': '0.4', 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '70', 'spark.default.parallelism': '180', 'spark.memory.storageFraction': '0.4' },
    # cost-metric-170.71123588085175 = 
 {'spark.executor.cores': '4', 'spark.executor.memory': '4g', 'spark.executor.instances': '6', 'spark.driver.cores': '1', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '72m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '80k', 'spark.broadcast.blockSize': '12m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.7', 'spark.rpc.message.maxSize': '224', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '90', 'spark.default.parallelism': '380', 'spark.memory.storageFraction': '0.7' }, 
    # cost-metric-170.71807479858398 =
    {'spark.executor.cores': '5', 'spark.executor.memory': '3g', 'spark.executor.instances': '6', 'spark.driver.cores': '1', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '64k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.6', 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'lz4', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '120', 'spark.default.parallelism': '140', 'spark.memory.storageFraction': '0.6' },
    # cost-metric-170.7461897134781 ^
    {'spark.executor.cores': '4', 'spark.executor.memory': '3g', 'spark.executor.instances': '5', 'spark.driver.cores': '4', 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '96m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '48k', 'spark.broadcast.blockSize': '18m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': '0.8', 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': '1', 'spark.sql.shuffle.partitions': '80', 'spark.default.parallelism': '260', 'spark.memory.storageFraction': '0.7' },
    # cost-metric-171.01545202732086 ^
    ]
    '''
    
    
    # random forest results
    RF_RESULTS = [{'spark.executor.cores': 2, 'spark.executor.memory': '8g', 'spark.executor.instances': 5, 'spark.driver.cores': 3, 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '32k', 'spark.broadcast.blockSize': '22m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': 0.4, 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': 1, 'spark.sql.shuffle.partitions': 120, 'spark.default.parallelism': 100, 'spark.memory.storageFraction': 0.5}, 
    {'spark.executor.cores': 6, 'spark.executor.memory': '7g', 'spark.executor.instances': 3, 'spark.driver.cores': 1, 'spark.driver.memory': '2g', 'spark.reducer.maxSizeInFlight': '56m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '48k', 'spark.broadcast.blockSize': '6m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': 0.7, 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'lz4', 'spark.task.cpus': 1, 'spark.sql.shuffle.partitions': 70, 'spark.default.parallelism': 300, 'spark.memory.storageFraction': 0.8},
    {'spark.executor.cores': 3, 'spark.executor.memory': '1g', 'spark.executor.instances': 6, 'spark.driver.cores': 4, 'spark.driver.memory': '4g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'false', 'spark.shuffle.file.buffer': '96k', 'spark.broadcast.blockSize': '14m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': 0.5, 'spark.rpc.message.maxSize': '256', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': 1, 'spark.sql.shuffle.partitions': 120, 'spark.default.parallelism': 260, 'spark.memory.storageFraction': 0.5},
    {'spark.executor.cores': 2, 'spark.executor.memory': '4g', 'spark.executor.instances': 7, 'spark.driver.cores': 2, 'spark.driver.memory': '2g', 'spark.reducer.maxSizeInFlight': '80m', 'spark.shuffle.compress': 'false', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '112k', 'spark.broadcast.blockSize': '4m', 'spark.broadcast.compress': 'false', 'spark.memory.fraction': 0.7, 'spark.rpc.message.maxSize': '192', 'spark.rdd.compress': 'true', 'spark.io.compression.codec': 'snappy', 'spark.task.cpus': 1, 'spark.sql.shuffle.partitions': 70, 'spark.default.parallelism': 380, 'spark.memory.storageFraction': 0.3},
    {'spark.executor.cores': 5, 'spark.executor.memory': '4g', 'spark.executor.instances': 6, 'spark.driver.cores': 3, 'spark.driver.memory': '3g', 'spark.reducer.maxSizeInFlight': '88m', 'spark.shuffle.compress': 'true', 'spark.shuffle.spill.compress': 'true', 'spark.shuffle.file.buffer': '48k', 'spark.broadcast.blockSize': '24m', 'spark.broadcast.compress': 'true', 'spark.memory.fraction': 0.7, 'spark.rpc.message.maxSize': '160', 'spark.rdd.compress': 'false', 'spark.io.compression.codec': 'lz4', 'spark.task.cpus': 1, 'spark.sql.shuffle.partitions': 120, 'spark.default.parallelism': 180, 'spark.memory.storageFraction': 0.5}
    ]
    
    param_lists = [deft, hand_tuning, hand_tuning10, rand_tuning] + MDP_RESULTS + RF_RESULTS
    param_lists = [make_param_list(pl) for pl in param_lists]
    return param_lists

def make_param_list(param_vals):
    '''
    given dictionary of param name -> chosen value,
    make a list of spark parameter dictionaries with those values
    without mutating original spark parameters
    '''
    params = []
    for p in sorted(SPARK_PARAMETERS, key = lambda x:x['name']):
        p = p.copy()
        p['cur_value'] = str(param_vals[p['name']])
        params.append(p)
    print(params)
    return params
    
    
### Examples running script
# python3 tpch_param.py 10

if __name__ == "__main__":
    try:
        N = int(sys.argv[1])
    except Exception as e:
        print(e)
        N = 10

    # make a unique log file name - list of all runtimes for sensitivity analysis
    '''
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_params/detparams_n{N}.json"
    
    with open(LOG_FNAME, "wb") as f:
        pass # create empty file? 

    print(f"starting deterministic list of params {N=}")
    deterministic_param_runs()
    print(f" {LOG_FNAME=} done")
    '''
    
    
    # make a unique log file name - comparing model / handtuning params against defaults
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_params/final_compare_params_n{N}.json"
    
    with open(LOG_FNAME, "wb") as f:
        pass # create empty file? 

    print(f"starting comparing list of params {N=}")
    
    param_lists = get_param_lists()
    comparison_param_runs(param_lists)
    print("# Param sets to compare", (len(param_lists)) )
    print(f" {LOG_FNAME=} done")
    
    

    