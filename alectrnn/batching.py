from abc import ABC, abstractmethod
import pickle
import uuid
import random


def generate_random_string():
    return str(uuid.uuid4())


def save(data, filename, protocol=pickle.DEFAULT_PROTOCOL):
    pickled_obj_file = open(filename, 'wb')
    pickle.dump(data, pickled_obj_file, protocol=protocol)
    pickled_obj_file.close()


def load(filename):
    pickled_obj_file = open(filename, 'rb')
    obj = pickle.load(pickled_obj_file)
    pickled_obj_file.close()
    return obj


class Batch(ABC):
    @abstractmethod
    def __call__(self):
        pass


class SeedBatch(Batch):
    def __init__(self, seed=None):
        random.seed(seed)
        self._current_seed = random.randint(0, 1000000)

    def __call__(self):
        seed = self._current_seed
        self._current_seed += 19
        return seed


def resolve_batches(parameters):
    resolved_parameters = {}
    for key, value in parameters.items():
        if isinstance(value, Batch):
            resolved_parameters[key] = value()
        elif isinstance(value, dict):
            resolved_parameters[key] = resolve_batches(value)
        else:
            resolved_parameters[key] = value

    return resolved_parameters


def generate_experiment_batch(num_trails,
                              experiment,
                              normalizer,
                              nervous_system_class,
                              nervous_system_parameters,
                              ale_parameters,
                              agent_parameters,
                              objective_parameters,
                              experiment_parameters,
                              trainer,
                              training_parameters,
                              cost_normalization_parameters,
                              batch_id=None,
                              storage_parameters=None):
    if storage_parameters is None:
        storage_parameters = {}

    if batch_id is None:
        batch_id = generate_random_string()

    parameter_batch = {'id': batch_id,
                       'batch': []}
    for _ in range(num_trails):
        parameter_batch['batch'].append(
            {
                'experiment': experiment,
                'normalizer': normalizer,
                'nervous_system_class': nervous_system_class,
                'nervous_system_parameters': resolve_batches(
                    nervous_system_parameters),
                'ale_parameters': resolve_batches(ale_parameters),
                'agent_parameters': resolve_batches(agent_parameters),
                'objective_parameters': resolve_batches(objective_parameters),
                'trainer': trainer,
                'training_parameters': resolve_batches(training_parameters),
                'cost_normalization_parameters': resolve_batches(
                    cost_normalization_parameters),
                'experiment_parameters': resolve_batches(
                    experiment_parameters),
                'storage_parameters': storage_parameters
            })

    return parameter_batch
