#!/usr/bin/python

import json
import math
import os
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime

import numpy as np
import logging

from util import MinHeap

log_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

program_log = logging.getLogger(__name__)
program_log.setLevel(logging.DEBUG)

file = logging.FileHandler("./logs/spark_tuner_mdp_%s.log" % log_suffix)
file.setLevel(logging.INFO)
file_format = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
file.setFormatter(file_format)

stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
stream_format = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", datefmt="%H:%M:%S")
stream.setFormatter(stream_format)

program_log.addHandler(stream)
program_log.addHandler(file)

# logging.basicConfig(filename="./logs/spark_tuner_mdp_%s.log" % log_suffix, encoding='utf-8', level=logging.DEBUG)

parser = ArgumentParser(description="MDP based spark tuning PoC")
parser.add_argument('--input',
                    '-i',
                    type=str,
                    default="sf10_training_data_long.json",
                    help="Path to input files for populating execute time matrix")
parser.add_argument('--params',
                    '-p',
                    type=str,
                    default="params_sf10.json",
                    help="Path to JSON config file for the parameters to be tuned")

# Global Variables
max_execution_time_placeholder = 0
sampled_states = set()
training_states = set()
extrapolated_states = dict()
extrapolated_states_average = dict()
near_states_map = dict()
param_candidates = MinHeap()


def parse_param_file(filepath):
  """
  Reads a JSON file of parameters to generate relevant data structures for tracking parameter information.

  :returns: list<param_label>, dictionary<param_label, index>,
            dictionary<index, valid_values>, dictionary<index, dictionary<value, value_index>>
  """
  if not os.path.isfile(filepath):
    raise Exception("Invalid filepath for parse_param_file operation. [filepath=%s]" % filepath)

  logging.info("Parsing parameter file [file=%s]" % filepath)
  with open(filepath, 'r') as f:
    data = json.load(f)

  param_label_map = {}
  param_valid_values_map = {}
  param_value_index_map = {}
  params = []
  counter = 0
  for param in data['params']:
    if not param['spark_param']:
      continue
    label = param['name']

    # Validations
    if label in param_label_map.keys():
      # ERROR: duplicate parameter value
      pass
    param_label_map[label] = counter
    param_value_index_map[counter] = {}
    valid_values = OrderedDict()
    for i in range(len(param['possible_values'])):
      valid_values[i] = param['possible_values'][i]
      param_value_index_map[counter][param['possible_values'][i]] = i
    params.append(label)
    param_valid_values_map[counter] = valid_values
    counter += 1
  return params, param_label_map, param_valid_values_map, param_value_index_map


def remove_outliers(numeric_list):
  q75, q25 = np.percentile(numeric_list, [75, 25])
  intr_qrtl = q75 - q25
  upper_bound = q75 + (1.5 * intr_qrtl)
  lower_bound = q25 - (1.5 * intr_qrtl)
  pruned_list = []
  for value in numeric_list:
    if lower_bound <= value <= upper_bound:
      pruned_list.append(value)

  return pruned_list


def parse_execution_params(execution_params, param_label_map, param_valid_values_map):
  parsed_params = [None for _ in range(len(param_label_map))]
  parsed_params_map = {}
  for param in execution_params:
    if not param['spark_param']:
      continue
    label = param['name']
    index = param_label_map[label]
    parsed_params[index] = param['cur_value']
    parsed_params_map[label] = param['cur_value']

  return parsed_params, parsed_params_map


def parse_input_file_newjson_reduced_maps(filepath, state_count, param_label_map,
                                          param_valid_values_map, param_index_base_map, param_value_index_map):
  global max_execution_time_placeholder
  # Validations
  if not os.path.isfile(filepath):
    raise Exception("Invalid filepath for parse_input_file operation. [filepath=%s]" % filepath)

  logging.info("Parsing sample data file [file=%s]" % filepath)
  reward_matrix = {}
  utility_list = {}

  with open(filepath, 'r') as f:
    data = json.load(f)

  for execution in data:
    runtimes = execution['runtimes']
    if 'total' not in runtimes.keys():
      # ERROR: Failed execution
      continue
    execution_time = runtimes['total']
    if type(execution_time) != int and type(execution_time) != float:
      # Add invalid parameter value combination to blacklist
      continue
    execution_time *= -1
    if execution_time < max_execution_time_placeholder:
      max_execution_time_placeholder = execution_time
    params, params_value_map = parse_execution_params(execution['params'],
                                                      param_label_map, param_valid_values_map)
    state_index = state_index_helper(params, param_index_base_map, param_value_index_map)
    reward_matrix[state_index] = execution_time
    utility_list[state_index] = execution_time
    sampled_states.add(state_index)
  return reward_matrix, utility_list


def parse_input_file_newjson(filepath, state_count, param_label_map,
                             param_valid_values_map, param_index_base_map, param_value_index_map):
  global max_execution_time_placeholder
  # Validations
  if not os.path.isfile(filepath):
    raise Exception("Invalid filepath for parse_input_file operation. [filepath=%s]" % filepath)

  reward_matrix = np.zeros(state_count)
  utility_list = np.zeros(state_count)

  with open(filepath, 'r') as f:
    data = json.load(f)

  for execution in data:
    runtimes = execution['runtimes']
    if 'total' not in runtimes.keys():
      # ERROR: Failed execution
      continue
    execution_time = runtimes['total']
    if type(execution_time) != int and type(execution_time) != float:
      # Add invalid parameter value combination to blacklist
      continue
    execution_time *= -1
    if execution_time < max_execution_time_placeholder:
      max_execution_time_placeholder = execution_time
    params, params_value_map = parse_execution_params(execution['params'],
                                                      param_label_map, param_valid_values_map)
    state_index = state_index_helper(params, param_index_base_map, param_value_index_map)
    reward_matrix[state_index] = execution_time
    utility_list[state_index] = execution_time
    sampled_states.add(state_index)
  return reward_matrix, utility_list


def parse_input_file(filepath, state_count, param_label_map,
                     param_valid_values_map, param_index_base_map, param_value_index_map):
  global max_execution_time_placeholder
  # Validations
  if not os.path.isfile(filepath):
    raise Exception("Invalid filepath for parse_input_file operation. [filepath=%s]" % filepath)

  reward_matrix = np.zeros(state_count)
  utility_list = np.zeros(state_count)

  with open(filepath, 'r') as f:
    data = json.load(f)

  for execution in data:
    runtimes = data[execution]['runtimes']
    if 'total' not in runtimes.keys():
      # ERROR: Failed execution
      continue
    execution_time = runtimes['total']
    if type(execution_time) != int and type(execution_time) != float:
      # Add invalid parameter value combination to blacklist
      continue
    execution_time *= -1
    if execution_time < max_execution_time_placeholder:
      max_execution_time_placeholder = execution_time
    params, params_value_map = parse_execution_params(data[execution]['params'],
                                                      param_label_map, param_valid_values_map)
    state_index = state_index_helper(params, param_index_base_map, param_value_index_map)
    reward_matrix[state_index] = execution_time
    utility_list[state_index] = execution_time
    sampled_states.add(state_index)
  return reward_matrix, utility_list


def perform_action_big(current_val_index, action_offset, param_index, param_valid_values_map, param_value_index_map):
  """
  NOTE:
    action-0: attempt to seek backwards one step over valid param value range,
    action-1: attempt to seek forwards one step over valid param value range,
  """
  valid_actions = (0, 1)
  # Validations
  if action_offset not in valid_actions:
    raise Exception("Invalid action offset")

  min_index = 0
  max_index = len(param_value_index_map[param_index])-1

  new_index = current_val_index
  if action_offset == 0 and new_index > min_index:
    new_index -= 1
  elif action_offset == 1 and new_index < max_index:
    new_index += 1
  return new_index


def perform_action(current_val, action_offset, param_index, param_valid_values_map, param_value_index_map):
  """
  NOTE:
    action-0: attempt to seek backwards one step over valid param value range,
    action-1: attempt to seek forwards one step over valid param value range,
  """
  valid_actions = (0, 1)
  # Validations
  if action_offset not in valid_actions:
    raise Exception("Invalid action offset")

  min_index = 0
  max_index = len(param_value_index_map[param_index])-1

  new_index = current_val
  if action_offset == 0 and new_index > min_index:
    new_index -= 1
  elif action_offset == 1 and new_index < max_index:
    new_index += 1

  return param_valid_values_map[param_index][new_index]


def transition_function_big(state_index, param_index, action_offset, param_valid_values_map,
                            param_index_base_map, param_value_index_map):
  # NOTE: 2 action per parameter dimension.
  # index/2 yields which parameter is being acted on.
  # index mod 2 yields which action is being taken.
  # Action 0: Iterate backwards (-1)
  # Action 1: Iterate forwards (+1)
  parameter_vals_indexes = state_values_helper(state_index, param_index_base_map, param_value_index_map)
  new_val = perform_action_big(parameter_vals_indexes[param_index], action_offset, param_index,
                               param_valid_values_map, param_value_index_map)
  parameter_vals_indexes[param_index] = new_val
  new_state_index = state_index_helper_indexes(parameter_vals_indexes, param_valid_values_map, param_index_base_map)
  return new_state_index


def generate_tuning_transition_matrix(parameter_count, state_count, param_valid_values_map,
                                      param_index_base_map, param_value_index_map):
  """
    parameter_count: total number of parameters being tuned for in config
    state_count: total number of possible states. Must be pre-computed and passed in.
    state_map: mapping from tuple of parameter values to a state index.
    state_list: list of parameter values represented by a state index.
  """

  # NOTE: 2 action per parameter dimension.
  # index/2 yields which parameter is being acted on.
  # index mod 2 yields which action is being taken.
  # Action 0: Iterate backwards (-1)
  # Action 1: Iterate forwards (+1)
  transition_matrix = [[] for _ in range(parameter_count*2)]
  for param_index in range(parameter_count):
    param_offset = param_index*2

    # Iterate over possible actions
    for action_offset in range(2):
      transition_matrix[param_offset+action_offset] = [np.zeros(state_count) for _ in range(state_count)]
      for state_index in range(state_count):
        # Retrieve parameter values represented by state_index
        parameter_vals = state_values_helper(state_index, param_index_base_map, param_value_index_map)
        # Perform action
        new_val = perform_action(parameter_vals[param_index], action_offset, param_index,
                                     param_valid_values_map, param_value_index_map)
        parameter_vals[param_index] = new_val
        new_state = state_index_helper(parameter_vals, param_index_base_map)
        transition_matrix[param_offset+action_offset][state_index][new_state] = 1.0

  return transition_matrix


def ordered_dictionary_get_index(ordered_dict, index):
  return ordered_dict.values()[index]


def vector_join(arr1, arr2):
  """
  every element of arr1 combined with every element of arr2
  """
  output = []
  for i in arr1:
    for j in arr2:
      output.append([i, j])

  return output


def vector_append(matrix1, arr2):
  """
  every element of arr2 combined with every element of matrix1
  """
  output = []
  for i in matrix1:
    for j in arr2:
      new_state = i.copy()
      new_state.append(j)
      output.append(new_state)

  return output


def evaluate_extrapolated_cost(state_index, extrapolated_times, dist_threshold=15.0, threshold_reduction=0.75):
  """
  dist_threshold is to have a weight mechanism on the extrapolated times as a way for closer extrapolations to be
  more impactful.
  """
  # (exec_time, rgv_magnitude)
  if state_index in extrapolated_states_average.keys():
    return extrapolated_states_average[state_index]
  sum = 0.0
  count = 0.0
  for exec_time, magnitude in extrapolated_times:
    ratio = min(1.0, float(magnitude)/dist_threshold)
    reduction = threshold_reduction*ratio
    factor = (1.0-reduction)
    sum += factor * exec_time
    count += factor * 1.0
  avg = float(sum)/count
  extrapolated_states_average[state_index] = avg
  return avg


def one_iteration_functional_pruned(state_count, parameter_count,
                                    utility_matrix, reward_matrix, gamma,
                                    param_valid_values_map, param_index_base_map, param_value_index_map):
  delta = 0
  for s in training_states:
    if s not in utility_matrix.keys():
      if s not in extrapolated_states.keys():
        utility_matrix[s] = max_execution_time_placeholder
      else:
        utility_matrix[s] = evaluate_extrapolated_cost(s, extrapolated_states[s])
      reward_matrix[s] = utility_matrix[s]
    temp = utility_matrix[s]
    v_list = np.zeros(parameter_count*2)
    for param_index in range(parameter_count):
      for action_offset in range(2):
        new_state_index = transition_function_big(s, param_index, action_offset, param_valid_values_map,
                                                  param_index_base_map, param_value_index_map)
        if new_state_index not in utility_matrix.keys():
          if new_state_index not in extrapolated_states.keys():
            utility_matrix[new_state_index] = max_execution_time_placeholder
          else:
            utility_matrix[new_state_index] = evaluate_extrapolated_cost(new_state_index,
                                                                         extrapolated_states[new_state_index])
          reward_matrix[new_state_index] = utility_matrix[new_state_index]
        util_matrix_val = utility_matrix[new_state_index]
        reward_matrix_val = reward_matrix[s]
        if reward_matrix_val == 0:
          reward_matrix_val = max_execution_time_placeholder
        action_index = (param_index*2)+action_offset
        v_list[action_index] = reward_matrix_val + gamma * util_matrix_val

    utility_matrix[s] = max(v_list)
    if len(param_candidates)-1 < 15:
      param_candidates.insert(utility_matrix[s], s)
    elif param_candidates.get_min()[0] < utility_matrix[s]:
      param_candidates.replace_min(utility_matrix[s], s)
    delta = max(delta, abs(temp - utility_matrix[s]))
  return delta


def one_iteration_functional(state_count, parameter_count,
                             utility_matrix, reward_matrix, gamma,
                             param_valid_values_map, param_index_base_map, param_value_index_map):
  delta = 0
  for s in range(state_count):
    temp = utility_matrix[s]
    if temp == 0:
      temp = max_execution_time_placeholder * 5
    # NOTE: Initialize influence value list. Represents utility of next state from a performed action.
    v_list = np.zeros(parameter_count*2)
    for param_index in range(parameter_count):

      # Iterate over possible actions
      for action_offset in range(2):
        # NOTES: New approach, using just a function for transition
        new_state_index = transition_function_big(s, param_index, action_offset, param_valid_values_map,
                                                  param_index_base_map, param_value_index_map)
        util_matrix_val = utility_matrix[new_state_index]
        if util_matrix_val == 0:
          util_matrix_val = max_execution_time_placeholder * 5
        reward_matrix_val = reward_matrix[s]
        if reward_matrix_val == 0:
          reward_matrix_val = max_execution_time_placeholder * 5
        action_index = (param_index*2)+action_offset
        v_list[action_index] = reward_matrix_val + gamma * util_matrix_val

    utility_matrix[s] = max(v_list)
    delta = max(delta, abs(temp - utility_matrix[s]))
  return delta


def one_iteration(state_count, action_count, state_list, state_map,
                  transition_matrix, utility_matrix, reward_matrix, gamma):
  delta = 0
  for s in range(state_count):
    temp = utility_matrix[s]
    # NOTE: Initialize influence value list. Represents utility of next state from a performed action.
    v_list = np.zeros(action_count)
    for a in range(action_count):
      p = transition_matrix[a][s]
      new_state_sum = 0
      for p_index in range(len(p)):
        new_state_sum += p[p_index]*utility_matrix[p_index]
      v_list[a] = reward_matrix[s] + gamma * new_state_sum

    utility_matrix[s] = max(v_list)
    delta = max(delta, abs(temp - utility_matrix[s]))
  return delta


def pruned_train(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                 param_valid_values_map, param_index_base_map, param_value_index_map, tol=1e-3):
  logging.info("Training initiated")
  epoch = 0
  delta = one_iteration_functional_pruned(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                                   param_valid_values_map, param_index_base_map, param_value_index_map)
  logging.info("Iteration completed [epoch=%d]" % epoch)
  delta_history = [delta]
  # return
  while delta > tol:
    epoch += 1
    delta = one_iteration_functional_pruned(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                                     param_valid_values_map, param_index_base_map, param_value_index_map)
    logging.info("Iteration completed [epoch=%d]" % epoch)
    delta_history.append(delta)
    if delta < tol:
      break
    if epoch >= 8:
      logging.info("Epoch 8 break out.")  # %
      break

  logging.info(f'# iterations of policy improvement: {len(delta_history)}')
  logging.info(f'delta = {delta_history}')


def uncertain_train(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                    param_valid_values_map, param_index_base_map, param_value_index_map, tol=1e-3):
  epoch = 0
  delta = one_iteration_functional(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                                   param_valid_values_map, param_index_base_map, param_value_index_map)
  delta_history = [delta]
  while delta > tol:
    epoch += 1
    delta = one_iteration_functional(state_count, parameter_count, utility_matrix, reward_matrix, gamma,
                                     param_valid_values_map, param_index_base_map, param_value_index_map)
    delta_history.append(delta)
    if delta < tol:
      break
    if epoch > 25:
      break


def train(state_count, action_count, state_list, state_map,
          transition_matrix, utility_matrix, reward_matrix, gamma, tol=1e-3):
  epoch = 0
  delta = one_iteration(state_count, action_count, state_list, state_map,
                        transition_matrix, utility_matrix, reward_matrix, gamma)
  delta_history = [delta]
  while delta > tol:
    epoch += 1
    delta = one_iteration(state_count, action_count, state_list, state_map,
                          transition_matrix, utility_matrix, reward_matrix, gamma)
    delta_history.append(delta)
    if delta < tol:
      break
    if epoch > 100:
      break

  logging.info(f'# iterations of policy improvement: {len(delta_history)}')
  logging.info(f'delta = {delta_history}')


def param_index_bases_helper(param_valid_values_map):
  """
  Helper for state_index generation.
  Help generate a map corresponding to required multiplication base depending on param_index.
  Current working formula: ... i*j_max*z_max + j*z_max + z*1 for 3 dimensions.
  """
  param_count = len(param_valid_values_map.items())
  param_index_base_mapping = {param_count-1: 1}
  # Iterate backwards from the "least significant" positioned param index
  for index in range(param_count-1, 0, -1):
    # index = param_count-i-1
    base = param_index_base_mapping[index]
    # Prepare next index base mapping
    param_index_base_mapping[index-1] = base*len(param_valid_values_map[index])

  return param_index_base_mapping


def state_index_helper_indexes(value_indexes, param_valid_values_map, param_index_base_map):
  """
  Converts from vector of parameter value indexes to the state index that represents that parameter value combination
  """
  state_index = 0
  for index in range(len(value_indexes)):
    value_index = value_indexes[index]
    # Validations
    if value_index is None:
      # ERROR: Parameter should not be None after proper parsing
      logging.error("Param is None [params=%s]" % (str(value_indexes)))
      raise Exception("Invalid")
    # Check index is valid
    if value_index not in param_valid_values_map[index].keys():
      # ERROR: Invalid parameter combination
      raise Exception("Invalid")

    state_index += value_index * param_index_base_map[index]
  return state_index


def state_index_helper(params, param_index_base_map, param_value_index_map):
  """
  Converts from vector of parameter values to the state index that represents that parameter value combination
  """
  state_index = 0
  for index in range(len(params)):
    param = params[index]
    if param is None:
      # ERROR: Parameter should not be None after proper parsing
      logging.error("Param is None [params=%s]" % (str(params)))
      continue
    relevant_param = param_value_index_map[index]
    value = relevant_param[param]
    state_index += value * param_index_base_map[index]
  return state_index


def param_index_helper(param_indexes, param_valid_values_map):
  """
  Converts from vector of parameter value indexes to the vector of parameter values represented by the indexes
  """
  params = [None for _ in range(len(param_indexes))]
  for i in range(len(param_indexes)):
    params[i] = list(param_valid_values_map[i].values())[param_indexes[i]]
  return params


def state_values_helper(state_index, param_index_base_map, param_value_index_map):
  """
  Converts from state index to vector of parameter value indexes represented by the state index
  """
  params = []
  remainder = state_index
  for i in range(len(param_index_base_map)):
    value_index = math.floor(remainder / param_index_base_map[i])
    remainder -= value_index * param_index_base_map[i]
    params.append(value_index)
  return params


def state_count_helper(param_valid_values_map):
  state_count = 1
  for value in param_valid_values_map.values():
    state_count *= len(value)
  return state_count


def format_candidate_output(params, param_labels, name="candidate_params"):
  output = name + " = {"
  prepend = ""
  for i in range(len(params)):
    output += prepend + "\'" + str(param_labels[i]) + "\': \'" + str(params[i]) + '\''
    prepend = ", "
  output += " }"
  return output


def generate_neighbors(state_index, param_valid_values_map, param_index_base_map,
                       param_value_index_map, add_probability=1.0):
  neighbors = set()
  for param_index in range(len(param_valid_values_map)):
    # Iterate over possible actions
    for action_offset in range(2):
      # NOTES: New approach, using just a function for transition
      if np.random.random_sample() <= add_probability:
        new_state_index = transition_function_big(state_index, param_index, action_offset, param_valid_values_map,
                                                  param_index_base_map, param_value_index_map)
        neighbors.add(new_state_index)

  return neighbors


def state_index_dist_helper(state1, state2, param_index_base_map, param_value_index_map):
  params1 = state_values_helper(state1, param_index_base_map, param_value_index_map)
  params2 = state_values_helper(state2, param_index_base_map, param_value_index_map)
  return math.dist(params1, params2)


def apply_param_vector(cur_params, reduced_gradient_vector, param_valid_values_map):
  is_changed = False
  new_params = [0 for _ in range(len(cur_params))]
  for i in range(len(cur_params)):
    cur_val = cur_params[i]
    delta = reduced_gradient_vector[i]
    new_val = cur_val
    if new_val + delta in param_valid_values_map[i].keys():
      # ERROR: Invalid parameter combination
      new_val += delta
      is_changed = True
    new_params[i] = new_val
  return new_params, is_changed


def generate_gradient_extrapolated_states(reward_matrix, param_valid_values_map,
                                          param_index_base_map, param_value_index_map):
  for item in near_states_map.items():
    cur_state = item[0]
    cur_exec_time = reward_matrix[cur_state]
    cur_params = np.array(state_values_helper(cur_state, param_index_base_map, param_value_index_map))
    for near_state in item[1]:
      index = near_state[0]
      near_exec_time = reward_matrix[index]
      # NOTE: Calculate impact of param delta on execute time
      exec_factor = float(cur_exec_time) / near_exec_time
      if not exec_factor < 1.0:
        continue
      near_params = np.array(state_values_helper(index, param_index_base_map, param_value_index_map))
      gradient_vector = cur_params - near_params
      gcd = np.gcd.reduce(gradient_vector)
      reduced_gradient_vector = gradient_vector/gcd
      # NOTE: Calculate ratio from original param delta to reduced gradient vector
      rgv_magnitude = np.linalg.norm(reduced_gradient_vector)
      gv_magnitude = np.linalg.norm(gradient_vector)
      ratio = rgv_magnitude / gv_magnitude

      extrapolated_params, is_changed = apply_param_vector(cur_params, reduced_gradient_vector, param_valid_values_map)
      if is_changed:
        extrapolated_state = state_index_helper_indexes(extrapolated_params,
                                                        param_valid_values_map,
                                                        param_index_base_map)
        if extrapolated_state in reward_matrix.keys():
          continue
        # NOTE: Cap extrapolated reductions at 20% (This should be enough to have relevant
        # extrapolated states be considered while also avoiding exrapolations from overshadowing
        # low execution time samples
        adjusted_reduction_factor = min(0.20, (1.0-exec_factor)*ratio)
        extrapolated_exec_time = cur_exec_time * (1.0 - adjusted_reduction_factor)
        element = (extrapolated_exec_time, rgv_magnitude)
        if extrapolated_state in extrapolated_states.keys():
          extrapolated_states[extrapolated_state].append(element)
        else:
          extrapolated_states[extrapolated_state] = [element]


def generate_near_sample_map(param_valid_values_map, param_index_base_map,
                             param_value_index_map, num_closest=100):
  logging.info("Populating sample-to-near-sample map")

  i = 0
  # NOTE: List of state_index format values
  indexable = list(sampled_states)
  while i < len(indexable):
    cur_state = indexable[i]
    near_samples = sorted(map(lambda sample: (sample,
                                       state_index_dist_helper(
                                         cur_state, sample, param_index_base_map, param_value_index_map)),
                              indexable), key=lambda sample: sample[1])[1:num_closest+1]
    near_states_map[cur_state] = near_samples
    i += 1


def generate_training_states(param_valid_values_map, param_index_base_map,
                             param_value_index_map, depth=1, probabilistic_depth=None):
  logging.info("Populating additional training states map from closest neighbors.")

  neighbors = set()
  training_states.update(sampled_states)
  cur_states = sampled_states
  cur_depth = 0
  while cur_depth < depth:
    next_neighbors = set()
    for s in cur_states:
      next_neighbors.update(generate_neighbors(s, param_valid_values_map, param_index_base_map, param_value_index_map))
    cur_states = next_neighbors
    neighbors.update(next_neighbors)
    cur_depth += 1
  if probabilistic_depth is not None:
    while cur_depth < probabilistic_depth:
      next_neighbors = set()
      for s in cur_states:
        next_neighbors.update(
          generate_neighbors(s, param_valid_values_map, param_index_base_map, param_value_index_map, 0.25))
      cur_states = next_neighbors
      neighbors.update(next_neighbors)
      cur_depth += 1
  training_states.update(neighbors)


def print_candidates(candidates, param_labels):
  sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
  output_str = "[\n"
  prepend = ""
  counter = 0
  for grouping in sorted_candidates:
    output_str += prepend + format_candidate_output(grouping[0], param_labels,
                                                    name=str(counter) + "-cost-metric" + str(grouping[1]))
    prepend = ",\n"
    counter += 1
  output_str += "\n]"
  print(output_str)


def run_model(param_filepath, data_filepath):
  global sampled_states
  param_labels, param_label_map, param_valid_values_map, param_value_index_map = parse_param_file(param_filepath)
  logging.info("[param_value_index_map=%s]" % str(param_value_index_map))
  param_index_base_map = param_index_bases_helper(param_valid_values_map)
  state_count = state_count_helper(param_valid_values_map)
  parameter_count = len(param_labels)

  reward_list, utility_list = parse_input_file_newjson_reduced_maps(
                                                  data_filepath, state_count, param_label_map,
                                                  param_valid_values_map, param_index_base_map, param_value_index_map)
  generate_near_sample_map(param_valid_values_map, param_index_base_map, param_value_index_map, num_closest=50)
  generate_gradient_extrapolated_states(reward_list, param_valid_values_map,
                                        param_index_base_map, param_value_index_map)
  training_states.update(sampled_states)
  training_states.update(extrapolated_states.keys())
  pruned_train(state_count, parameter_count, utility_list, reward_list, 0.5,
               param_valid_values_map, param_index_base_map, param_value_index_map)
  optimal_candidates = {}
  for state_index, execution_time in param_candidates.candidates_time.items():
    param_indexes = state_values_helper(state_index, param_index_base_map, param_value_index_map)
    optimal_candidates[tuple(param_index_helper(param_indexes, param_valid_values_map))] = execution_time
  print_candidates(optimal_candidates, param_labels)


if __name__ == "__main__":
  args = parser.parse_args()
  logging.info("LOGGING: Start Spark Config Tuning Model")
  run_model(args.params, args.input)
