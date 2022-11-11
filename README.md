1. connect vpn
2. `ssh haoyuw@general-cirrascale-15.reviz.ai2.in`
3. `cd SRL_to_TemporalGraph`
4. make changes to the code
5. `bash create_beaker_nonm1.sh`
6. `beaker group create my_big_parallel_experiment_x`
7. `beaker experiment create -n temprel_run_x.xx beaker_example.yaml` or `nohup python3 my_big_parallel_experiment.py > mbpe.out 2>&1 &`
8. check status at https://beaker.org/experiments

`python demo.py <model> <input> <output>`

`CUDA_VISIBLE_DEVICES=0 nohup python demo.py ./ /shared/corpora-tmp/annotated_nyt/156 ./output > 156.out 2>&1 &`
