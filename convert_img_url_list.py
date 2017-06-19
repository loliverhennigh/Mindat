

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

with open('all_minerals.csv', 'r') as f:
  all_minerals = f.readlines()[0].replace(' ','').split(',')

# split url out
for i in xrange(len(lines)):
  new_line = lines[i].split(',')
  replace_line = []
  replace_line.append(new_line[0])
  for j in xrange(len(new_line)-2):
    mineral_name = _convert_string_to_mineral_list(new_line[j+1])
    if mineral_name in all_minerals:
      replace_line.append(mineral_name)
  lines[i] = replace_line 

img_url_list_converted_file = open("img_url_list_converted.csv", "w")
for l in lines:
  if len(l) == 1:
    print(l)
    continue
  for v in l:
    if v != ' ':
      img_url_list_converted_file.write(v + ', ')
  img_url_list_converted_file.write('\n')
  


