import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Parameter:
    def __init__(self, name):
        self.name = name
        self.val_to_results = {}

    def add(self, param_val, results):
        self.val_to_results[param_val] = results

    def get_total(self, param_val):
        if param_val in self.val_to_results:
            return self.val_to_results[param_val]["total"]

    def get_param_vals(self):
        return set(self.val_to_results.keys())

param_name_to_param_obj = {}

def populate_params():
    #analyze one file
    direc = "./training_data/training_results/"
    files = os.listdir(direc)
    files = [f for f in files if os.path.isfile(direc+'/'+f)] #just files
    for file in files:
        if "deterministic" in file:
            f = open(direc + file)
            data = json.load(f)
            print(file, len(data))
            for k in data:
                d = data[k]
                params = d["params"]
                runtimes = d["runtimes"]
                if len(runtimes) == 0:
                    continue

                all_default = True
                for p in params:
                    if p['spark_param'] and p['cur_value'] != p['default_value']:
                        varying_param = p['name']
                        all_default = False
                        break
                if all_default:
                    param_name_to_param_obj["default"] = runtimes
                    continue
                if varying_param not in param_name_to_param_obj:
                    param_name_to_param_obj[varying_param] = Parameter(varying_param)
                obj = param_name_to_param_obj[varying_param]
                assert "total" in runtimes
                obj.add(p['cur_value'], runtimes)
            f.close()
            return

def analyze_results():
    populate_params()
    default = param_name_to_param_obj["default"]
    default_total = default["total"]
    mins = []
    names = []
    diffs = []
    stds = []
    for p in param_name_to_param_obj:
        min_runtime = float("inf")
        max_runtime = -float("inf")
        times = []
        if p != 'default':
            obj = param_name_to_param_obj[p]
            for k in obj.get_param_vals():
                total_time = obj.get_total(k)
                times.append(total_time)
                min_runtime = min(min_runtime, total_time)
                max_runtime = max(max_runtime, total_time)
            
        else:
            times.append(default_total)
            min_runtime = default_total
            max_runtime = default_total + 0.001
        stds.append(np.std(times))
        print(f"{p}, min: {min_runtime}, max: {max_runtime}, default: {default_total}")
        mins.append(min_runtime)
        diffs.append(max_runtime - min_runtime)
        names.append(p)

    #save total time std to csv
    df = pd.DataFrame(list(zip(names, stds)),
    columns =['name', 'standard_dev_total_times'])
    df.to_csv("./tables/time_stds.csv", mode='w')

    #plot std
    fig = plt.figure(figsize = (10, 5))
    plt.xticks(rotation=90)
    plt.bar(names, stds, bottom=[0]*len(names), width=0.2)
    plt.title("Standard deviation of total query times for each parameter")
    plt.xlabel("Parameter")
    plt.ylabel("Standard deviation of times")
    plt.savefig("./plots/time_stds.jpg", bbox_inches="tight")

    #plot min/max times for each parameter
    fig = plt.figure(figsize = (10, 5))
    plt.xticks(rotation=90)
    plt.bar(names, diffs, bottom=mins, width=0.2)
    plt.title("Range of total query times for each parameter")
    plt.xlabel("Parameter")
    plt.ylabel("Range of times")
    plt.savefig("./plots/time_ranges.jpg", bbox_inches="tight")
    plt.show()

analyze_results()