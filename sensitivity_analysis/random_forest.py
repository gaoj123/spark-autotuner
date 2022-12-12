from unittest.util import sorted_list_difference
from sklearn.ensemble import RandomForestRegressor
from training_data.tpch_param import SPARK_PARAMETERS
import json
import os
from skopt.space import Categorical
import pandas as pd
import numpy as np
from skopt import plots, gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import matplotlib.pyplot as plt

ORDER_OF_PARAMS = []

for i in range(len(SPARK_PARAMETERS)):
    name = SPARK_PARAMETERS[i]["name"]
    ORDER_OF_PARAMS.append(name)

class TrainingData:
    def __init__(self):
        self.param_to_val = {}
        self.total_time = None

    def add(self, param, val):
        self.param_to_val[param] = val

    def set_total_time(self, t):
        self.total_time = t

    def get_total_time(self):
        return self.total_time

    def get_val(self, param):
        return self.param_to_val.get(param, None)

def get_training_data():
    direc = "./training_data/training_results/"
    files = os.listdir(direc)
    files = [f for f in files if os.path.isfile(direc+'/'+f)] #just files
    x = []
    y = []
    for file in files:
        if "sf10_training_data.json" in file: 
            f = open(direc + file)
            data = json.load(f)
            for k in data:
                d = data[k]
                params = d["params"]
                runtimes = d["runtimes"]
                if len(runtimes) == 0:
                    continue

                curr_x = []
                for p in params:
                    if p['spark_param']:
                        curr_x.append(p['cur_value'])
                assert "total" in runtimes
                
                total_time = runtimes['total']
                if total_time < 400: #remove outliers
                    x.append(curr_x)
                    y.append(total_time )

    print(f"len x: {len(x)}, len y: {len(y)}")
    return x, y

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Model:
    def __init__(self):
        self.model = None
        self.all_cols = None

    def build_model(self):
        x, y = get_training_data() #, order_of_params = get_training_data()
        df = pd.DataFrame(x, columns=ORDER_OF_PARAMS) #, columns=ORDER_OF_PARAMS)
        print(len(df.columns))
        df2 = pd.get_dummies(df, columns=ORDER_OF_PARAMS) #CATEGORICAL_PARAMS)
        print(len(df2.columns))
        X_train, X_test, y_train, y_test = train_test_split(df2, y,
                                        test_size=0.2, 
                                        shuffle=True)

        self.all_cols = list(df2.columns)

        regr = RandomForestRegressor(max_depth=2, random_state=0)
        self.model = regr.fit(X_train, y_train) #X, y)

        score = self.model.score(X_test, y_test)
        print(f"SCORE: {score}")
        print(len(X_train), len(X_test), len(x))

        y = self.model.predict(X_test)
     
        loss = np.sqrt(mean_squared_error(y_test, y))
        print(f"LOSS: {loss}")
        mape = mean_absolute_percentage_error(y_test, y)
        print(f"MAPE: {mape}")

        param_imp = {}
        feature_imp = regr.feature_importances_
        for i in range(len(feature_imp)):
            param = self.all_cols[i] 
            param_imp[param] = feature_imp[i]
        sorted_param_imp =  {k: v for k, v in sorted(param_imp.items(), key=lambda item: item[1], reverse=True)}
        print(sorted_param_imp)

        aggregated_param_imp = defaultdict(int)
        for p in sorted_param_imp:
            name = p.split("_")[0]
            aggregated_param_imp[name] += sorted_param_imp[p]
        
        sorted_agg_param_imp =  {k: v for k, v in sorted(aggregated_param_imp.items(), key=lambda item: item[1], reverse=True)}
        print(f"sorted aggregated param imp: {sorted_agg_param_imp}")


    def objective_func(self, x, **kwargs):
        data = pd.DataFrame([x], columns=ORDER_OF_PARAMS)
        data_to_categorical= pd.get_dummies(data, columns=ORDER_OF_PARAMS#CATEGORICAL_PARAMS
               ).reindex(columns=self.all_cols).fillna(0).astype('int')
        row = data_to_categorical.iloc[0]
        predicted_time = self.model.predict([row])
        return predicted_time[0]

    def bayesian_optimization(self):
        pbounds = [] 
        for i in range(len(SPARK_PARAMETERS)):
            pbounds.append(Categorical(SPARK_PARAMETERS[i]["possible_values"]))

        num_iters = 100
        num_init_points = 30

        best_set = {}
        best_time = float("inf")

        for run in range(5):
            print(f"START RUN {run}")
            optimizer = gp_minimize(func=self.objective_func, dimensions=pbounds, n_calls=num_iters, n_initial_points=num_init_points, acq_func="LCB", kappa=5000)

            for i in range(len(optimizer.x_iters)):
                print(f"Iter {i}: {optimizer.x_iters[i]}, val: {optimizer.func_vals[i]}")
            print(f"Best: {optimizer.x}, val: {optimizer.fun}")


            best_parameters = {}
            for i, p in enumerate(ORDER_OF_PARAMS):
                best_parameters[p] = optimizer.x[i]

            if optimizer.fun < best_time:
                best_time = optimizer.fun
                best_set = best_parameters
            
            print(f"Curr best parameter set: {best_parameters}, time: {optimizer.fun}")
            fig = plt.figure(figsize = (10, 5))
            ax = plots.plot_convergence(optimizer)
            plt.savefig(f"./plots/bayesian_opt_convergence_{run}.jpg", bbox_inches="tight")
        print(f"Overall best set: {best_set}, time: {best_time}")
        plt.show()

if __name__ == "__main__":
    m = Model()
    m.build_model()
    m.bayesian_optimization()

