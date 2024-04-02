Install dependencies:
pip install -r requirements.txt

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

