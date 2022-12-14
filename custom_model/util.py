import math


class MinHeap:
  def __init__(self, collection=None):
    # Initialize a heap using list of elements (<queue_size>, <uuid>)
    self.heap = [[-math.inf, -1]]
    self.FRONT = 1
    self.candidates = {}
    self.candidates_time = {}
    if collection is not None:
      self.heap = collection

  def __len__(self):
    return len(self.heap)-1

  def get_parent_position(self, i):
    # The parent is located at floor((i-1)/2)
    # return int((i - 1) / 2)
    return i // 2

  def get_left_child_position(self, i):
    # The left child is located at 2 * i + 1
    # return 2 * i + 1
    return 2 * i

  def get_right_child_position(self, i):
    # The right child is located at 2 * i + 2
    # return 2 * i + 2
    return (2 * i) + 1

  def has_parent(self, i):
    # This function checks if the given node has a parent or not
    return self.get_parent_position(i) < (len(self.heap)-1)

  def is_leaf(self, i):
    return i*2 > (len(self.heap)-1)

    # Function to swap two nodes of the heap
  def swap(self, fpos, spos):
    self.heap[fpos], self.heap[spos] = self.heap[spos], self.heap[fpos]

  def has_left_child(self, i):
    # This function checks if the given node has a left child or not
    return self.get_left_child_position(i) < len(self.heap)

  def has_right_child(self, i):
    # This function checks if the given node has a right child or not
    return self.get_right_child_position(i) < len(self.heap)

  def insert(self, key, state_index):
    # if len(self.heap)-1 >= self.maxsize:
    #   return
    # self.size += 1
    self.heap.append([key, state_index])
    if key in self.candidates.keys():
      self.candidates[key].add(state_index)
    else:
      self.candidates[key] = {state_index}
    self.candidates_time[state_index] = key

    current = len(self.heap) - 1

    while self.heap[current][0] < self.heap[self.get_parent_position(current)][0]:
      self.swap(current, self.get_parent_position(current))
      current = self.get_parent_position(current)

  def replace_min(self, key, state_index):
    replaced_index = 1
    if state_index in self.candidates_time.keys():
      old_time = self.candidates_time[state_index]
      if len(self.candidates[old_time]) > 1:
        self.candidates[old_time].remove(state_index)
      else:
        self.candidates.pop(old_time)
      if key in self.candidates.keys():
        self.candidates[key].add(state_index)
      else:
        self.candidates[key] = {state_index}
      self.candidates_time[state_index] = key
      for i in range(len(self.heap)):
        if self.heap[i][1] == state_index:
          self.heap[i][0] = key
          replaced_index = i
          break
    else:
      old_time = self.heap[1][0]
      if len(self.candidates[old_time]) > 1:
        self.candidates[old_time].remove(self.heap[1][1])
      else:
        self.candidates.pop(old_time)
      self.candidates_time.pop(self.heap[1][1])
      self.heap[1] = [key, state_index]
      if key in self.candidates.keys():
        self.candidates[key].add(state_index)
      else:
        self.candidates[key] = {state_index}
      self.candidates_time[state_index] = key
    self.min_heap()

  def min_heap(self):
    for pos in range((len(self.heap)) // 2, 0, -1):
      self.heapify(pos)

  def get_min(self):
    # return self.heap[0]  # Returns the lowest value in the heap in O(1) time.
    return self.heap[1]  # Returns the lowest value in the heap in O(1) time.

  def heapify(self, i):
    # If the node is a non-leaf node and greater
    # than any of its child
    if not self.is_leaf(i):
      if (self.heap[i][0] > self.heap[self.get_left_child_position(i)][0]
              or (self.has_right_child(i) and self.heap[i][0] > self.heap[self.get_right_child_position(i)][0])):

        # Swap with the left child and heapify
        # the left child
        if not self.has_right_child(i) \
                or self.heap[self.get_left_child_position(i)][0] < self.heap[self.get_right_child_position(i)][0]:
          self.swap(i, self.get_left_child_position(i))
          self.heapify(self.get_left_child_position(i))

        # Swap with the right child and heapify
        # the right child
        else:
          self.swap(i, self.get_right_child_position(i))
          self.heapify(self.get_right_child_position(i))

  def print_heap(self):
    print(self.heap)