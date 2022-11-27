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
make_param('spark.sql.shuffle.partitions', True, 200, list(range(100,1001, 100))),
make_param('spark.default.parallelism', True, 200, list(range(20,400, 40))),
make_param('spark.memory.storageFraction', True, .5, [x/10 for x in list(range(3,9))]), 
#make_param('spark.ui.port', True, 4040, [i for i in range(4040, 4058)]),
]

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
    for i in range(10):
        log_results(params)
    
    for param in params:
        if param['spark_param']:
            for val in param['possible_values']:
                if val != param['default_value']:
                    param['cur_value'] = val
                    for i in range(10):
                        log_results(params)

            # reset param back to default
            param['cur_value'] = param['default_value']
    

                    
def randomize_params():
    new_params = []
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
python3 – version 
pip3 install --user numpy
pip3 install --user pyspark

python3  ../../../../../../../spark-autotuner/training_data/tpch_training.py main_job_n4_mem-per-cpu4000

'''

if __name__ == "__main__":

    # get number of files 
    direc = f"{CURRENT_FILE_PATH}/training_params/"
    files = os.listdir(direc)
    files = [f for f in files if os.path.isfile(direc+'/'+f)] #just files
    num_files = len(files)
    
    # make a unique log file name - list of all runtimes for sensitivity analysis
    LOG_FNAME = f"{CURRENT_FILE_PATH}/training_params/detparams_{num_files+1}.json"
    
    with open(LOG_FNAME, "wb") as f:
        pass # create empty file? 

    print("starting deterministic list of params")
    deterministic_param_runs()
    print("done")

