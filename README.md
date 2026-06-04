# Wasserstein Policy Optimization (WPO) Toy Examples

## Numerical example for mixture of Gaussians.

Run the following command to run the example:

```bash
python run_toy_example.py --method PG --batch-size 1024 --lr 0.005 --num-iters 500 
python run_toy_example.py --method NPG --batch-size 1024 --lr 0.005 --num-iters 500 
python run_toy_example.py --method WPO --batch-size 1024 --lr 0.005 --num-iters 500

python vis_toy_example.py
```