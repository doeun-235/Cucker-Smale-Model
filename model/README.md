## Model
In the codes, we deal with 4 models; Cucker-Smale model, the model introduced in [[3]][(CKPP19)] and its extension for more general networks, and the model introduced in [[2]][(O21)]. The first three of them are systems of ordinary differential equation and the last one is a system of stochastic differential equation.

For 3 ODE models, I used Runge-Kutta 4th method, well-known as 'ode45' in MATLAB. I simulated the SDE model, the main model of my thesis, with an improved Euler-Maruyama method proposed in [[4]][(R12)]. I coded them with NumPy.  

## Bibliography

1. [CS07][(CS07)] : Felipe Cucker and Steve Smale. Emergent behavior in flocks. IEEE Transactions
on Automatic Control, 52(5):852–862, 2007.

2. [O21][(O21)] : Tackgeun Oh, Flocking Behavior in Stochastic Cucker-Smale Model with Formation Control on Symmetric Digraphs, Yonsei Univ., 2021.
/ ( I changed my name from 'Tackgeun Oh' to 'Doeun Oh'. )

3. [CKPP19][(CKPP19)] : Young-Pil Choi, Dante Kalise, Jan Peszek, and Andrés A. Peters. A collisionless
singular Cucker-Smale model with decentralized formation control. SIAM J.
Appl. Dyn. Syst., 18(4):1954–1981, 2019.

4. [R12][(R12)] : A. J. Roberts. Modify the improved euler scheme to integrate stochastic differential equations, preprint, arXiv:1210.0933.

## License

MIT

[(CS07)]: https://ieeexplore.ieee.org/document/4200853 "CS07"
[(O21)]: http://www.riss.kr/link?id=T15771814 "O21"
[(CKPP19)]: https://arxiv.org/abs/1807.05177 "CKPP19"
[(R12)] : https://arxiv.org/abs/1210.0933 "R12"
