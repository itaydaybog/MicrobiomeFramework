import gzip
import numpy as np


def get_results_path(selection_type, fitness_decay, relevant_step):
    if selection_type == 'decay_fitness':
        return f'{selection_type}_{str(fitness_decay)}'
    elif selection_type == 'step_fitness':
        return f'{selection_type}_{str(relevant_step)}'
    else:  # neutral_fitness'
        return selection_type


def get_transmission_name(parent_prob, peers_prob, env_prob):
    if parent_prob == 1 and peers_prob == 0 and env_prob == 0:
        return 'vertical'
    elif parent_prob == 0 and peers_prob == 1 and env_prob == 0:
        return 'horizontal'
    elif parent_prob == 1 and peers_prob == 1 and env_prob == 0:
        return 'midway'
    else:
        return f'custom_{parent_prob}_{peers_prob}_{env_prob}'


def save_single_result_as_npy_gz(path, result):
    with gzip.GzipFile(path, "w") as f:
        np.save(file=f, arr=result)
