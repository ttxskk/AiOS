
from detrsmpl.apis.test import (
    collect_results_cpu,
    collect_results_gpu,
    multi_gpu_test,
    single_gpu_test,
)
from detrsmpl.apis.train import set_random_seed, train_model

__all__ = [
    'collect_results_cpu', 'collect_results_gpu','multi_gpu_test','single_gpu_test'
]
