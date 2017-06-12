

from Queue import Queue
from threading import Thread

import urllib2
import re

from tqdm import *

# Work queue where you push the URLs onto - size 100
url_queue = Queue(10)
pattern = re.compile('"></a>(.+?)</dd><dt>')

list_of_links = []

def worker():
    '''Gets the next url from the queue and processes it'''
    while True:
        url = url_queue.get()
        print url
        html = urllib2.urlopen(url).read()
        print html[:10]
        list_of_links.append(html[:10])
        links = pattern.findall(html)
        if len(links) > 0:
            print links
        url_queue.task_done()

# Start a pool of 20 workers
for i in xrange(20):
     t = Thread(target=worker)
     t.daemon = True
     t.start()

# Change this to read your links and queue them for processing
for url in tqdm(xrange(30)):
    url_queue.put("http://www.ravn.co.uk")

# Block until everything is finished.
url_queue.join()   

print(list_of_links)

