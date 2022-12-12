# Spark Autotuner

Hope Dargan, Jenny Gao, Gabriel Jimenez, Min Thet Khine

## Random Forest and Bayesian Optimization
Running the command "python3 sensitivity_analysis/random_forest.py" will train the random forest model using the data we have (stored in a JSON file).  After training, it will output the feature importance (Gini importance).  The random forest model is also used as a cost estimator for the Bayesian optimization search, which seeks to find the set of parameters with the lowest query time.  This code will run Bayesian optimization five times, each with 100 iterations and 30 initial points.