

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
    print(string)
    for i in xrange(len(string)):
      if string[i] == '(':
        start = i
    string = string[:start]
    print(string)

  return string 

with open('img_url_list.csv', 'r') as f:
  lines = f.readlines()

# split url out
all_minerals = []
all_minerals_count = dict()
for i in xrange(len(lines)):
  new_line = lines[i].split(',')
  for j in xrange(len(new_line)-1):
    new_line[j+1] = _convert_string_to_mineral_list(new_line[j+1])
  for j in xrange(len(new_line)-2):
    if new_line[j+1] not in all_minerals:
      all_minerals.append(new_line[j+1])
      all_minerals_count[new_line[j+1]] = 1
    else:
      all_minerals_count[new_line[j+1]] += 1
  lines[i] = new_line

for i in xrange(1000):
  print(lines[i])
all_minerals.sort()
print(all_minerals)
print(all_minerals_count)
print(len(all_minerals))
total_count = 0
for m in all_minerals:
  if all_minerals_count[m] > 200:
    print(m)
    print(all_minerals_count[m])
    total_count += 1
    #total_count += all_minerals_count[m]
print(total_count)
    

