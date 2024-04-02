# MetaPhysiCa: Improving OOD Robustness in Physics-informed Machine Learning

This repository is the official implementation of [MetaPhysiCa: Improving OOD Robustness in Physics-informed Machine Learning](https://openreview.net/forum?id=KrWuDiW4Qm)

> Abstract: A fundamental challenge in physics-informed machine learning (PIML) is the design of robust PIML methods for out-of-distribution (OOD) forecasting tasks. These OOD tasks require learning-to-learn from observations of the same (ODE) dynamical system with different unknown ODE parameters, and demand accurate forecasts even under out-of-support initial conditions and out-of-support ODE parameters. In this work we propose to improve the OOD robustness of PIML via a meta-learning procedure for causal structure discovery. Using three different OOD tasks, we empirically observe that the proposed approach significantly outperforms existing state-of-the-art PIML and deep learning methods (with 2 to 28 times lower OOD errors).


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiments
Run MetaPhysiCa:
python metaphysica.py --data=damped_pendulum --datatype=2

`--data` can be "damped_pendulum", "lotka_volterra" or "sir".
`--datatype=2` for OOD initial condition X0.
`--datatype=3` for OOD initial condition X0 and OOD parameters W*.

Additional options:
  --polynomial_power=p              Max power of polynomial basis [default: 3]
  --lr=lr                           Learning rate [default: 1e-2]
  --lambda_phi=r1                   L1 regularization strength [default: 1e-2]
  --lambda_vrex=r2                  V-REx penalty strength [default: 0]
