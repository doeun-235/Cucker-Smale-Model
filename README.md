# Simulations for Cucker-Smale-Model and Its Extensions

## Introduction

In this repository, I did simulations about the Cucker-Smale model [[1]][(CS07)], second-order-nonlinear ODEs describing the flocking behaviors, and its extensions. Mostly, the works have been done for writing my master's degree thesis; [[2]][(O21)].

In the directory named 'model', there are two directories, 'ODE' and 'SDE'. In the directory 'ODE', there are simulation codes for solving ODE systems, Cucker-Smale model, Cucker-Smale model with decentralized formation control term in [[3]][(CKPP19)], and an extension of the model in [[3]][(CKPP19)] on more general graphs. In the directory 'SDE', there are simulation codes for solving the SDE system introduced in [[2]][(O21)], which has a formation controller and multiplicative noises.

The main goals of codes in directories 'ODE' and 'SDE' are simulating ODE and SDE systems mentioned in a preceding paragraph and showing that solutions converge when initial conditions satisfy specific conditions. The main features implemented in the codes are visualizing movements of agents as video and showing numerical results support the theoretical results in [[1]][(CS07)], [[3]][(CKPP19)] and [[2]][(O21)] with plots. 

Also, I made a code in the directory 'etc' for visualizing networks and it is used for writing [[2]][(O21)]. In the directory 'figure', there are video files and plots made by the codes. Aimed patterns through simulations were <img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\pi" title="\pi" /> shaped pattern and Einstein's face image pattern. I get the curve equations for the shapes from https://wolframalpha.com.

More specific instructions are in each directory.

## Bibliography

1. [CS07][(CS07)] : Felipe Cucker and Steve Smale. Emergent behavior in flocks. IEEE Transactions
on Automatic Control, 52(5):852–862, 2007.

2. [O21][(O21)] : Tackgeun Oh, Flocking Behacior in Stochastic Cucker-Smale Model with Formation Control on Symmetric Digraphs, Yonsei Univ., 2021.
/ ( I changed my name from 'Tackgeun Oh' to 'Doeun Oh'. )

3. [CKPP19][(CKPP19)] : Young-Pil Choi, Dante Kalise, Jan Peszek, and Andrés A. Peters. A collisionless
singular Cucker-Smale model with decentralized formation control. SIAM J.
Appl. Dyn. Syst., 18(4):1954–1981, 2019.

## License

MIT

[(CS07)]: https://ieeexplore.ieee.org/document/4200853 "CS07"
[(O21)]: http://www.riss.kr/link?id=T15771814 "O21"
[(CKPP19)]: https://arxiv.org/abs/1807.05177 "CKPP19"
