# Spark Autotuner

Hope Dargan, Jenny Gao, Gabriel Jimenez, Min Thet Khine

## TPC-H for Generating Training Data

### Generating TPC-H Tables

The repo contains the `TPC-H V3.0.1` folder that was downloaded from the TPC website. To generate the TPC-H tables, the given `makefile` in `TPC-H V3.0.1/dbgen` must first be configured depending on the platform according to the instructions in `makefile.suite`. Note that our repo already contains a version of the `makefile` that can produce TPC-H tables in a platform-agnostic manner. 

To generate the TPC-H tables:
- Create the data generator script by running:
```
make
```
- Generate the TPC-H tables (default scale factor: 1) by running:
```
./dbgen
```
Larger scale factors can be speciifed by giving the scale factor with a `-s` option:
```
./dbgen -s 100
```

The above command should produce TPC-H tables in `TPC-H V3.0.1/dbgen` in `.tbl` format.

### Generating TPC-H Queries

The `TPC-H V3.0.1/dbgen` folder contains all the necessary TPC-H queries in `TPC-H V3.0.1/dbgen/queries`. Note that the starting queries that are downloaded are just templates and do not have exact values. We substituted the placeholders with the recommended benchmarking values as mentioned in the TPC-H specification. We also converted query 15 to use a Common Table Expression (CTE) since PySpark expects a single SQL statement.

## Collecting Training Data

* convert_tables.py
The training_data has all files needed to generate and analyze training data.
After generating the TPC-H tables and queries first convert the .tbl files to parquet files. 
`$ python3 convert_tables.py '/PATH/TO/spark-autotuner/tpch_tables/s1' 1`
Extra args are output destination and table scale factor. Note this output folder is the assumed TPC-H table location in the other scripts that run the queries and log the runtimes.

* tpch_param.py

To collect data deterministically, you will need to first generate .json files with the parameter combinations. 
To generate a json files that iterates through all possible parameter values N times, run
`$ python3 tpch_param.py 10` 
extra argument is N.

This file was also used for generating parameter combinations to run the final comparisons (as evidenced by the get_param_lists function and commented out code at the bottom.) You can also set the parameters and possible values you want to explore by modifying the SPARK_PARAMETERS list in tpch_param.py.

* tpch_det.py

To actually collect deterministic data, run 
`$ python3 tpch_det.py test 1 10 ` 
Note the additional args are job name, table scale factor, and number of runs for each parameter. This can easily be changed by updating the `if __name__ == '__main__'` section of each python script.
This will run all the TPC-H queries once on the next parameter set and log the results in a file that depends on all three arguments. Re-running the same command will cause training data to accumulate in the same folder. All slurm jobs submitted to the cluster were essentially just 800 lines running the same script over and over again until either the job timed out or we ran out of parameters in the parameter json file. (see old_scripts/big_det_slurm_job.txt for an example of this monstrosity.) Because of this one parameter run per script execution <i>feature</i> it is best to collect data via submitting jobs to a computing cluster.

* tpch_compare.py

tpch_compare.py does the same thing as tpch_det.py but points to a different parameter file. We created a separate file so we could run both regular deterministic training runs and comparison runs in parallel on the cluster, but we probably should have just had the tpch_det script take in the parameter json file path.
`$ python3 tpch_compare.py test 1` 
Note extra args are job name and table scale factor.


* tpch_rand.py

This generates a random parameter set, runs the TPC-H queries it, and adds the resulting runtime to the file indicated by the jobname and scale factor.

`$ python3 tpch_rand.py test 1`
Additional args are job name and scale factor.


* training_analysis.ipynb

The training_analysis.ipynb file is a hodge podge of small scripts that do a variety of things. The function of each cell is described, but this file is what we used to generate training data statistics, compile parallel training data files into a single json that can be fed to the models, make the pretty graphs, etc.

The old_scripts and old_training_results folders are historical artifacts which preserve old code and training results in all their original broken glory.  

The queries folder contains a copy of the generated TPC-H queries.

The training_params folder contains the json files used for deterministic runs.

The training_results folder contains the raw text files created by deterministic and random runs.

The training_sensitivity folder contains many graphs generated from every stage of training data.

Note all the python scripts liberally import from each other, so they should be stored in the same directory.

## Random Forest and Bayesian Optimization
Running the command "python3 sensitivity_analysis/random_forest.py" will train the random forest model using the data we have (stored in a JSON file).  After training, it will output the feature importance (Gini importance).  The random forest model is also used as a cost estimator for the Bayesian optimization search, which seeks to find the set of parameters with the lowest query time.  This code will run Bayesian optimization five times, each with 100 iterations and 30 initial points.

## Customized MDP Model
The python project for the customized MDP model is contained in the `custom_model` directory.
To run the model against our training data execute:
```
# NOTE: Assumption that python3 points to python 3 interpreter
# NOTE: Dependency on 'numpy' python package 
python3 ./custom_model/custom_model.py
```
The model will run against the data stored in the JSON file `/custom_model/sf10_training_data_long.json`, tuning for the parameters specified in `/custom_model/params_sf10.json`.

Logs will be contained in the `/custom_model/logs` directory. After processing and training, should observe a print-out of the candidate parameter combinations for low execution times.
