# Wasserstein Policy Optimization (WPO) Toy Examples

## Numerical example for mixture of Gaussians.

Run the following command to run the example:

```bash
python run_toy_example.py --method PG --batch-size 1024 --lr 0.005 --num-iters 500 
python run_toy_example.py --method NPG --batch-size 1024 --lr 0.005 --num-iters 500 
python run_toy_example.py --method WPO --batch-size 1024 --lr 0.005 --num-iters 500

python vis_toy_example.py
```

## Inverted pendulum example.

DDPG-style actor-critic experiment for Gymnasium `Pendulum-v1`, with three actor update options:

- `PG`: stochastic Gaussian actor with log-likelihood gradients weighted by critic values.
- `DPG`: deterministic policy gradient / DDPG-style actor update.
- `WPO`: stochastic Gaussian actor with the WPO action-gradient transport update and simplified Gaussian Fisher scaling.


```bash
python train.py --method PG  --num-iters 30000 --gaussian-fisher-scaling wpo --seed 42
python train.py --method DPG --num-iters 30000 --gaussian-fisher-scaling wpo --seed 42
python train.py --method WPO --num-iters 30000 --gaussian-fisher-scaling wpo --seed 42
```

Results are written to:

```text
./results/inverted_pendulum/<run_name>/
```

Each run contains `config.json`, `metrics.csv`, `actor.pt`, and `critic.pt`.

To visualize one run or compare multiple runs under a parent folder:

```bash
python visualize.py --result-dir ./results/inverted_pendulum
```