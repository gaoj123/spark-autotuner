from sklearn.ensemble import RandomForestRegressor
import json
import os

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
    #analyze one file
    direc = "./training_data/old_training_results/"
    files = os.listdir(direc)
    files = [f for f in files if os.path.isfile(direc+'/'+f)] #just files
    x = []
    y = []
    training_data = []
    for file in files:
        if "deterministic" in file:
            f = open(direc + file)
            data = json.load(f)
            print(file, len(data))
            order_of_params = []
            got_order = False
            for k in data:
                d = data[k]
                params = d["params"]
                runtimes = d["runtimes"]
                if len(runtimes) == 0:
                    continue

                if not got_order:
                    for p in params:
                        if p['spark_param']:
                            order_of_params.append(p['name'])
                    print(f"order of params: {order_of_params}")
                    print(f"number of params: {len(order_of_params)}")
                    got_order = True

                t = TrainingData()
                for p in params:
                    if p['spark_param']:
                        val = None
                        try:
                            val = float(p['cur_value'])
                        except:
                            val = p['possible_values'].index(p['cur_value'])
                        t.add(p['name'], val)
                assert "total" in runtimes
                t.set_total_time(runtimes['total'])
                training_data.append(t)
                
                curr_x = []
                for p in order_of_params:
                    curr_x.append(t.get_val(p))
                x.append(curr_x)
                y.append(t.get_total_time())
              
            f.close()
            print(f"len x: {len(x)}, len y: {len(y)}")
            return x, y, training_data, order_of_params

def model():
    x, y, training_data, order_of_params = get_training_data()
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    regr.fit(x, y)

    param_imp = {}
    feature_imp = regr.feature_importances_
    for i in range(len(feature_imp)):
        param = order_of_params[i]
        param_imp[param] = feature_imp[i]
    sorted_param_imp =  {k: v for k, v in sorted(param_imp.items(), key=lambda item: item[1], reverse=True)}

    #print(feature_imp)
    #print(param_imp)
    print(f"sorted param importance {sorted_param_imp}")
    print(f"params sorted by importance: {list(sorted_param_imp.keys())}")

model()