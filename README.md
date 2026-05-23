# evolutionaryalgo

Code written before Jan 2021 but was deleted from the main spikey repo so I add it here.

Commit where its been removed from Spikey: https://github.com/SpikeyCNS/spikey/commit/5368bfd4eda060aa3e4d4accb5da4c45aa24e9af

## 8-Queens demo

Run [run_8queens.py](run_8queens.py) to evolve a solution to the 8-queens
problem using `MetaNQueens` + `Population` (200 agents/epoch, single-process
backend, seed 0). Fitness is `28 - clashes`; the game terminates when a
genotype with 0 clashes is found.

### Results

- Solved at **epoch 73** with fitness **28** (no queens attacking).
- **Total genotypes tested (incl. cache hits): 14,800**
- Unique genotypes evaluated (cache misses): 9,893
- Search space size: $8^{16} \approx 2.8 \times 10^{14}$ (each of 8 queens has independent x and y in [0, 7]).

### Final board

```
. . . Q . . . .
. . . . . . Q .
. . . . Q . . .
. . Q . . . . .
Q . . . . . . .
. . . . . Q . .
. . . . . . . Q
. Q . . . . . .
```

Best genotype:

```
a=(6,6)  b=(5,2)  c=(0,3)  d=(2,4)
e=(3,7)  f=(7,1)  g=(4,5)  h=(1,0)
```


### Final boards across seeds

Generated with `python run_8queens.py -q 0 2 3 4`:

```
    seed=0             seed=2             seed=3             seed=4      
. . . Q . . . .    . . . Q . . . .    . Q . . . . . .    Q . . . . . . .
. . . . . . Q .    . Q . . . . . .    . . . . . Q . .    . . . . Q . . .
. . . . Q . . .    . . . . . . . Q    Q . . . . . . .    . . . . . . . Q
. . Q . . . . .    . . . . . Q . .    . . . . . . Q .    . . . . . Q . .
Q . . . . . . .    Q . . . . . . .    . . . Q . . . .    . . Q . . . . .
. . . . . Q . .    . . Q . . . . .    . . . . . . . Q    . . . . . . Q .
. . . . . . . Q    . . . . Q . . .    . . Q . . . . .    . Q . . . . . .
. Q . . . . . .    . . . . . . Q .    . . . . Q . . .    . . . Q . . . .
```
