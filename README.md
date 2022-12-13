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

## Random Forest and Bayesian Optimization
Running the command "python3 sensitivity_analysis/random_forest.py" will train the random forest model using the data we have (stored in a JSON file).  After training, it will output the feature importance (Gini importance).  The random forest model is also used as a cost estimator for the Bayesian optimization search, which seeks to find the set of parameters with the lowest query time.  This code will run Bayesian optimization five times, each with 100 iterations and 30 initial points.
