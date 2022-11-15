import json
import requests

def temporal_getter(SRL_output, onepass = 1):
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
    
with open('files.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        with open(line) as srl:
            srl_result = json.load(srl)
            srl_result['folder'] = line
            print(temporal_getter(srl_result))
            

            
