# Code for [Extracting or Guessing? Improving Faithfulness of Event Temporal Relation Extraction](https://arxiv.org/pdf/2210.04992.pdf)

## Environment
### Method 1: 
`conda activate EMNLP_env`
### Method 2: 
`source /shared/why16gzl/conda/miniconda38/etc/profile.d/conda.sh` <br>
`conda activate EMNLP_env`
#### Note that the environment is at '/shared/why16gzl/conda/miniconda38/envs/EMNLP_env'
#### You can create a same environment using environment_droplet.yml

## Feb 6, 2023:
### backend.py: a CherryPy API; accepts raw text as input; running on holst 6011
### How to run the backend service:
`nohup python backend.py > backend.out 2>&1 &`
### Example curl command: 
`curl -d '{"text": "The Supreme Court rejected earlier appeals for a case review from five of the convicts."}' -H "Content-Type: application/json" -X POST localhost:6011/annotate`
#### response of the command above: 
{"status": "Success", "elasped_time": "0:00:03", "output_timeline": {"6": {"event_pairs": [[1, 0], [0, 2], [1, 2]], "conf": [0.9358216115822632, 0.6843115049770331, 0.8746635341117158], "event_pairs_pwa": [[1, 0]], "conf_pwa": [0.9358216115822632], "timeline_topo_sort_pwa": [[1, 0]]}}, "event_info": "--> 0 Event: 'reject' (0_0_3)\tARG0: 'The Supreme Court' (NA, 0_0_0)\tARG1: 'earlier appeals for a case review from five of the convicts' (NA, 0_0_4)\t\n--> 1 Event: 'appeal' (0_0_5)\tARG2: 'The Supreme Court' (NA, 0_0_0)\tSupport: 'rejected' (NA, 0_0_3)\tARGM-TMP: 'earlier' (NA, 0_0_4)\tARG1: 'for a case review from five of the convicts' (NA, 0_0_6)\t\n--> 2 Event: 'review' (0_0_9)\tARG1: 'case' (NA, 0_0_8)\tARG0: 'from five of the convicts' (NA, 0_0_10)\t\n"}

## Nov 15, 2022:
## backend_nov15.py: a CherryPy API; accepts SRL-processed NYTimes document (stored in /shared/corpora-tmp/annotated_nyt) as input; running on holst 6012

### How to run the backend service:
`nohup python backend_nov15.py > backend_nov15.out 2>&1 &`

#### A) if the input is an SRL-processed NYTimes document, for example, sample.json:
`curl --data @sample.json -H "Content-Type: application/json" -X POST http://127.0.0.1:6012/annotate`
##### Response to the command above (note that the system can return more than one timeline since there can be multiple possible ßtopological sorting results):
{"status": "Success", "elasped_time": "0:00:08", "text": "JERRY SEINFELD is moving away from Tom's Restaurant in more ways than one. First he signed a $4.35 million contract for an apartment a couple of blocks south of his old one on Central Park West. Then NBC's soon-to-depart sitcom star dropped plans to film one of his final episodes in New York. The show will end its run on a California sound stage.\nThe deal for the apartment (now owned by ISAAC STERN) hinges on approval from the co-op board at the Beresford, at 211 Central Park West.\n''No one's particularly worried -- the building already has a celebrity quotient,'' said one person familiar with the deal, noting that his new neighbors would include HELEN GURLEY BROWN, SIDNEY LUMET, TONY RANDALL and BEVERLY SILLS. But would Kramer still be able to wander in?\nPUBLIC LIVES", "events": {"0": "move", "1": "sign", "2": "drop", "3": "end", "4": "own", "5": "have", "6": "say", "7": "note", "8": "include", "9": "be", "10": "wander"}, "output_timeline": {"/shared/corpora-tmp/annotated_nyt/156/995130.ta.xml": {"event_pairs": [[1, 0], [2, 0], [0, 3], [4, 0], [5, 0], [0, 6], [0, 7], [0, 8], [0, 9], [10, 0], [1, 2], [1, 3], [1, 4], [5, 1], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [2, 3], [4, 2], [5, 2], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [4, 3], [5, 3], [3, 6], [3, 7], [3, 8], [9, 3], [10, 3], [5, 4], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [5, 6], [5, 7], [5, 8], [5, 9], [5, 10], [6, 7], [6, 8], [6, 9], [10, 6], [7, 8], [7, 9], [10, 7], [8, 9], [10, 8], [10, 9]], "conf": [0.9560927918292985, 0.5178540295264422, 0.8769560447412562, 0.9194619577654775, 0.8962519287582138, 0.8828677729219634, 0.8119130117425453, 0.8015191476880698, 0.6899372465437733, 0.5350570457240086, 0.9132374821879031, 0.8993119849662943, 0.701176429437062, 0.4806056220253727, 0.9092900549523073, 0.8792367098207637, 0.9379764939684672, 0.9079443412913253, 0.8690610020531173, 0.9417460485584712, 0.4700946832291738, 0.8301165562913201, 0.9215896493352291, 0.891086845621458, 0.8895155682284969, 0.9015244375925091, 0.8151547547980024, 0.9551522753361968, 0.9555983200336572, 0.7025524561885114, 0.5906448154535038, 0.537950826540171, 0.49302685422709464, 0.8133303345068548, 0.6154362888254176, 0.9115646522123139, 0.8978601897181051, 0.9223000058992876, 0.9168039610716654, 0.8803068739809811, 0.8991615644633276, 0.9203687085069268, 0.8982641183030375, 0.8954294194638307, 0.8803484841695854, 0.5483668828244295, 0.6388949126698036, 0.5331965738800951, 0.8111576806362703, 0.6153571179147675, 0.6432124876287697, 0.8245966297692827, 0.5523616850089162, 0.7109230201038257, 0.812588251690407], "event_pairs_pwa": [[1, 0], [4, 0], [5, 0], [1, 2], [1, 3], [1, 6], [1, 8], [1, 9], [2, 3], [2, 6], [2, 9], [4, 3], [5, 3], [4, 6], [4, 7], [4, 8], [4, 9], [5, 6], [5, 7], [5, 8], [5, 9]], "conf_pwa": [0.9560927918292985, 0.9194619577654775, 0.8962519287582138, 0.9132374821879031, 0.8993119849662943, 0.9092900549523073, 0.9379764939684672, 0.9079443412913253, 0.9417460485584712, 0.9215896493352291, 0.9015244375925091, 0.9551522753361968, 0.9555983200336572, 0.9115646522123139, 0.8978601897181051, 0.9223000058992876, 0.9168039610716654, 0.8991615644633276, 0.9203687085069268, 0.8982641183030375, 0.8954294194638307], "timeline_topo_sort_pwa": [[5, 1, 4, 2, 0, 9, 3, 6, 7, 8], [5, 1, 4, 2, 0, 6, 8, 9, 3, 7], [5, 1, 4, 2, 0, 6, 7, 9, 3, 8], [5, 1, 4, 2, 0, 6, 7, 8, 9, 3], [5, 1, 4, 2, 0, 6, 9, 3, 7, 8], [5, 1, 4, 2, 0, 3, 6, 7, 8, 9], [5, 1, 4, 2, 0, 8, 9, 3, 6, 7], [5, 1, 4, 2, 0, 7, 9, 3, 6, 8], [5, 1, 4, 2, 0, 7, 8, 9, 3, 6]]}}}

#### B) if you want to process multiple SRL-processed NYTimes documents, call run.py whose input is a .txt file (e.g., samples.txt) that specifies the document paths for each .xml file. Note that if the processing time of a document is longer than 3 minutes, the system aborts it. How to run: `python run.py --files samples.txt`
##### Note that the results are stored in the folder "/shared/corpora-tmp/nyt_event_temporal_graph/" (we create a corresponding folder for each NYT folder, e.g., 156); intermediate files are stored in "/shared/cache/"


## Nov 11, 2022

## demo.py: accept a folder that contains srl results as input; output to a json file that contains all the event temporal graphs. 
### Problem: some files are too big so that either taking too much time to run or taking too much GPU memory -> error

`python demo.py <model> <input> <output>`

`CUDA_VISIBLE_DEVICES=0 nohup python demo.py ./ /shared/corpora-tmp/annotated_nyt/156 ./output > 156.out 2>&1 &`

## Implement using AI2 beaker 
1. connect vpn
2. `ssh username@general-cirrascale-15.reviz.ai2.in`
3. `cd SRL_to_TemporalGraph`
4. make changes to the code
5. `bash create_beaker_nonm1.sh`
6. `beaker group create my_big_parallel_experiment_x`
7. `beaker experiment create -n temprel_run_x.xx beaker_example.yaml` or `nohup python3 my_big_parallel_experiment.py > mbpe.out 2>&1 &`
8. check status at https://beaker.org/experiments
