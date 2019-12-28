from abc import ABC, abstractmethod
import pickle
import uuid


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


def resolve_batches(parameters):
    resolved_parameters = {}
    for key, value in parameters:
        if isinstance(value, Batch):
            resolved_parameters[key] = value()
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
                              cost_normalization_parameters):
    parameter_batch = {'id': generate_random_string(),
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
                'agent_parameters': agent_parameters,
                'objective_parameters': resolve_batches(objective_parameters),
                'trainer': trainer,
                'training_parameters': training_parameters,
                'cost_normalization_parameters': resolve_batches(
                    cost_normalization_parameters),
                'experiment_parameters': experiment_parameters
            })

    return parameter_batch
