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

### Scalability analysis
- Let problem size be number of vertices $v$
- number of $v$ can be obrained from the *static method* GraphManager.get_graph_files()


Finding the population $n_{req}$ required to solve a problem 10/10 times
- find $n_{upper}$ using n = [2, 4, 6 , 8 ...]
- Let $n_{lower} = \frac{n_{upper}}{2}$
- Search $n_{req}$ between $n_{lower}$ and $n_{upper}$ by increasing $n$ by a factor of ***1.1*** each time

- Over 10 different instances of the graph with same number of $v$, take the mean $\mu_{n_{req}}$ and the deviation $\pm \mu_{n_{req}} = \frac{max(n_{req})- min (n_{req})}{2}$

- **Save csv with columns `set_name`, `v`, `n_req_mean`,  `n_req_max`, and `n_req_min`**
- Then plot graph with `v` on the x-axis and `n_req_mean` on y-axis`


