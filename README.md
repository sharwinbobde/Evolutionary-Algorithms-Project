# Evolutionary Algorithms Project (Group ??)
CS4205 Evolutioanry Algorithms
members :
- Sharwin Bobde (5011639)
- Thalis Papakyriakou
- Isha Dijcks
- Rickard Hellström


## Instructions 
save the maxcut instances under `/data/maxcut`. Example of a `.txt` file relative path would be `data/maxcut/set0a/n0000006i00.txt`


## Methodology

### Finding `n_req` Scalability analysis
- Let problem size be number of vertices $v$
- number of $v$ can be obrained from the *static method* GraphManager.get_graph_files()


Finding the population $n_{req}$ required to solve a problem 10/10 times
- find $n_{upper}$ using n = [2, 4, 6 , 8 ...]
- Let $n_{lower} = \frac{n_{upper}}{2}$
- Search $n_{req}$ between $n_{lower}$ and $n_{upper}$ by increasing $n$ by a factor of ***1.1*** each time



## Model Comparisons
We will be doing the comparison for both Black-box and Grey/White-box approach.
Save all observations in csv as following:
  - filename `<EA-used>-<B or W><metric name>-<set_name>.csv` 
  - with the columns `v`, `<metric name>_mean`,  `<metric name>_std`
  - Examples: 
    - `particle_swarm-B-num_eval-set0b.csv` with columns `v`, `num_eval_mean`, `num_eval_std`
    - `particle_swarm-W-runtime-set0a.csv` with columns `v`, `runtime_mean`, `runtime_std`
    - `particle_swarm-B-gen-set0c.csv` with columns `v`, `gen_mean`, `gen_std`

### Metrics
- number of fitness evaluations (mean and std. for 10 runs of 5 graphs per problem set) (`num_eval`)
- runtime in sec. (mean and std. for 10 runs of 5 graphs per problem set)   (`runtime`)
- Np. of generations. (mean and std. for 10 runs of 5 graphs per problem set)   (`gen`)

