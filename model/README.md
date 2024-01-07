## Model
In the codes, we deal with 4 models; Cucker-Smale model[[1]][(CS07)], the model introduced in [[3]][(CKPP19)] and its extension for more general networks, and the model introduced in [[2]][(O21)]. The first three of them are systems of ordinary differential equation and the last one is a system of stochastic differential equation.

The first one, the Cucker-Smale model, describes movement of a particle system, all of them converges to same velocity. The model is written as 

\begin{equation}
    \begin{array}
        \frac{\text{d}}{\text{d}t}x_t^i & = & v_t^i, \quad i=1,\cdots,N, \quad t>0 \\
        \frac{\text{d}}{\text{d}t}v_t^i & = &  K \sum_{j=1}^{N}\psi (\vert x_t^j-x_t^i \vert)(v_t^j-v_t^i)
    \end{array}
\end{equation}

```math 
\begin{equation}
    \begin{array}
        \frac{\text{d}}{\text{d}t}x_t^i & = & v_t^i, \quad i=1,\cdots,N, \quad t>0 \\
        \frac{\text{d}}{\text{d}t}v_t^i & = &  K \sum_{j=1}^{N}\psi (\vert x_t^j-x_t^i \vert)(v_t^j-v_t^i)
    \end{array}
\end{equation}
```


$a_i$

```c
void main
```

```math
a_i
```


## Bibliography

1. [CS07][(CS07)] : Felipe Cucker and Steve Smale. Emergent behavior in flocks. IEEE Transactions
on Automatic Control, 52(5):852–862, 2007.

2. [O21][(O21)] : Tackgeun Oh, Flocking Behavior in Stochastic Cucker-Smale Model with Formation Control on Symmetric Digraphs, Yonsei Univ., 2021.
/ ( I changed my name from 'Tackgeun Oh' to 'Doeun Oh'. )

3. [CKPP19][(CKPP19)] : Young-Pil Choi, Dante Kalise, Jan Peszek, and Andrés A. Peters. A collisionless
singular Cucker-Smale model with decentralized formation control. SIAM J.
Appl. Dyn. Syst., 18(4):1954–1981, 2019.

## License

MIT

[(CS07)]: https://ieeexplore.ieee.org/document/4200853 "CS07"
[(O21)]: http://www.riss.kr/link?id=T15771814 "O21"
[(CKPP19)]: https://arxiv.org/abs/1807.05177 "CKPP19"
