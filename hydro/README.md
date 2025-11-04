The file **sod_exact.csv** contain the exact solution for the Sod problem at t = 0.2 s, using Riemann solver from [Toro (2009)](https://link.springer.com/book/10.1007/b79761) (see Chap. 4 ) and gamma = 1.4

To access the file use:

```bash
import pandas as pd
sod = pd.read_csv('sod_exact.csv', index_col=0)
```
