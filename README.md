# Nov 15, 2022:
## backend.py: cherrypy api; accept srl result as input; write event temporal graph (.etg file) to /shared disk

## How to run the backend service:
`nohup python backend.py > backend.out 2>&1 &`

## The files to be processed is in "files.txt"

## How to process the files (building event temporal graphs for them):
`python run.py`

## The results are stored in the folder "/shared/corpora-tmp/nyt_event_temporal_graph/"

## The intermediate files are stored in "/shared/cache/"

# Nov 11, 2022

## demo.py: accept a folder that contains srl results as input; output to a json file that contains all the event temporal graphs. 
## Problem: some files are too big so that either taking too much time to run or taking too much GPU memory -> error

`python demo.py <model> <input> <output>`

`CUDA_VISIBLE_DEVICES=0 nohup python demo.py ./ /shared/corpora-tmp/annotated_nyt/156 ./output > 156.out 2>&1 &`

## Implement using AI2 beaker 
1. connect vpn
2. `ssh haoyuw@general-cirrascale-15.reviz.ai2.in`
3. `cd SRL_to_TemporalGraph`
4. make changes to the code
5. `bash create_beaker_nonm1.sh`
6. `beaker group create my_big_parallel_experiment_x`
7. `beaker experiment create -n temprel_run_x.xx beaker_example.yaml` or `nohup python3 my_big_parallel_experiment.py > mbpe.out 2>&1 &`
8. check status at https://beaker.org/experiments

