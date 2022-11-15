import json
import requests
from tqdm import tqdm
import time
import stopit
import multiprocessing as mp
import os

def temporal_getter(line):
    with open(line) as srl:
        SRL_output = json.load(srl)
        SRL_output['folder'] = line
        
    headers = {'Content-type':'application/json'}
    temporal_service = 'http://localhost:6010/annotate'
    print("Calling service from " + temporal_service)
    temporal_response = requests.post(temporal_service, json=SRL_output, headers=headers)
    
    if temporal_response.status_code != 200:
        print("temporal_response:", temporal_response.status_code)
    try: 
        result = json.loads(temporal_response.text)
        return result
    except:
        return {"status": "failure"}
    
def timeout(func, args = (), kwds = {}, timeout = 1, default = None):
    pool = mp.Pool(processes = 1)
    result = pool.apply_async(func, args = args, kwds = kwds)
    try:
        val = result.get(timeout = timeout)
    except mp.TimeoutError:
        pool.terminate()
        return default
    else:
        pool.close()
        pool.join()
        return val
    
with open('/shared/why16gzl/Repositories/EventCausalityData/files.txt') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line[:-1]
        print("Processing " + line)
        if os.path.exists("/shared/corpora-tmp/nyt_event_temporal_graph/" + line.split('/')[-2] + '/' + line.split('/')[-1] + ".etg"):
            print("Already processed this file")
            continue
        timeout(temporal_getter, args = (line,), timeout = 60*3, default = 'Bye')  
        
        if not os.path.exists("/shared/corpora-tmp/nyt_event_temporal_graph/" + line.split('/')[-2] + '/' + line.split('/')[-1] + ".etg"):
            print("$$$$$$$$$$$$$$$$ " + line + " TIMEOUT!")
        
        #with stopit.ThreadingTimeout(60*3) as context_manager:
        #    with open(line) as srl:
        #        srl_result = json.load(srl)
        #        srl_result['folder'] = line
        #        print(temporal_getter(srl_result))
        #if context_manager.state == context_manager.EXECUTED:
        #    print("COMPLETE...")
        #elif context_manager.state == context_manager.TIMED_OUT:
        #    print("DID NOT FINISH... " + line)
        #    continue
        
        



 
            

            
