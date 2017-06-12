

from io import StringIO, BytesIO
import requests
from lxml import etree
import os
import sys
import imghdr
from tqdm import *
from Queue import Queue
from threading import Thread

# read list of img urls
with open('img_url_list.csv', 'r') as f:
  lines = f.readlines()
url_list = []
for l in lines:
  url_list.append(l.split(',')[0]) 

# make worker
url_queue = Queue(10)
def worker():
  while True:
    url = url_queue.get()
    img_data = requests.get(url).content
    name = '/data/mindat-images/' + '_'.join(url.split('/')[3:])
    if not os.path.exists(name):
      with open(name, 'wb') as handler:
        handler.write(img_data)
    else:
    url_queue.task_done()
   
for i in xrange(20):
  t = Thread(target=worker)
  t.daemon = True
  t.start()

# get all 900,000 urls
for url in tqdm(url_list):
  url_queue.put(url)
url_queue.join()
