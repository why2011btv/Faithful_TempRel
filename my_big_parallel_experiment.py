import subprocess
import os
onlyfiles = []
with open('dir_name.txt') as file:
    lines = file.readlines()
    for line in lines:
        onlyfiles.append(line[:-1])
        
gen_spec = """description: first beaker run with the TempRel code
tasks:
  - name: first_TempRel_run
    image:
      beaker: haoyuw/haoyuw-6046c23_1_cuda111_1
    arguments: [python, demo.py, /model, /input, /output]
    envVars:
      - name: MY_VAR
        value: "hello"
    datasets:
      - mountPath: /model
        source:
          beaker: haoyuw/0511pm.pt
      - mountPath: /input
        source:
          beaker: haoyuw/%s
    result:
      path: /output
    resources:
      gpuCount: 1
    context:
      priority: preemptible
    constraints:
      cluster:
        - ai2/aristo-cirrascale"""

def execute_beaker(spec,name,group):
    temp_out = "./yaml_spec.yml"
    with open(temp_out,'w') as my_yaml:
        print(spec,file=my_yaml)
    p = subprocess.Popen(
        "beaker experiment create -n %s %s" % (name,temp_out),
        shell=True
    )
    p.wait()
    os.remove(temp_out)
    if group is not None: 
        b = subprocess.Popen(
            "beaker group add %s haoyuw/%s" % (group,name),
            shell=True
        )
        b.wait()
        
for dir_name in onlyfiles:
    spec = gen_spec % ('NYT_' + dir_name)
    name = 'exp_4_' + 'NYT_' + dir_name
    group = 'haoyuw/my_big_parallel_experiment_4'
    execute_beaker(spec,name,group)
    
    
