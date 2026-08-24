"""Wrapper: init Ray with explicit GPUs, then run miles train.py.

Ray must see the GPUs that SLURM allocated via CUDA_VISIBLE_DEVICES.
We init Ray BEFORE importing miles so ray.init(address="auto") is a no-op.
"""
import sys
import ray

ray.init(num_gpus=4, num_cpus=32, ignore_reinit_error=True)
print(f"Ray resources: {ray.cluster_resources()}")

# Run train.py as __main__ so its if __name__ == "__main__" block fires.
# sys.argv[0] must point at train.py for miles' arg parser (it checks argv[0]).
sys.argv[0] = "/proj/assert-berzelius/users/x_tuhan/garage/miles/train.py"
with open(sys.argv[0]) as f:
    code = compile(f.read(), sys.argv[0], "exec")
    exec(code, {"__name__": "__main__"})
