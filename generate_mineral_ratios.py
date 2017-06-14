

import os
import sys
from tqdm import *

# helper function
def _convert_string_to_mineral_list(string):
  string = string.lower()
  string = string.replace(' ','')
  start = -1 
  end = -1 
  if 'var' in string:
    for i in xrange(len(string)):
      if string[i] == '(':
        start = i
    string = string[:start]

  return string 

with open('img_url_list.csv', 'r') as f:
  lines = f.readlines()

# split url out
all_minerals = []
all_minerals_count = dict()
sum_minerals = 0.0
for i in xrange(len(lines)):
  new_line = lines[i].split(',')
  for j in xrange(len(new_line)-1):
    new_line[j+1] = _convert_string_to_mineral_list(new_line[j+1])
  for j in xrange(len(new_line)-2):
    if new_line[j+1] not in all_minerals:
      all_minerals.append(new_line[j+1])
      all_minerals_count[new_line[j+1]] = 1
      sum_minerals += 1
    else:
      all_minerals_count[new_line[j+1]] += 1
      sum_minerals += 1
  lines[i] = new_line
all_minerals.sort()
    

all_minerals_file = open("minerals_ratios.py", "w")
all_minerals_file.write('mineral_ratios = [')
number_of_mins = 0
for m in all_minerals:
  if all_minerals_count[m] > 300:
    if m != '':
      all_minerals_file.write(str(all_minerals_count[m]/sum_minerals) + ', ')
      number_of_mins += 1
print(number_of_mins)

