# Instantiate configured classes and convert nested configuration keys (modified for Mallamar).
from argparse import Namespace
from collections.abc import MutableMapping, Mapping
from typing import Type, Dict, Any
import importlib
import copy
import inspect
import math, os
from numbers import Integral, Real


def flatten_namespace(ns, parent_key='', separator='.'):
    items = {}
    for key, value in ns.__dict__.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, Namespace):
            items.update(flatten_namespace(value, new_key, separator=separator))
        elif isinstance(value, dict):
            items.update(flatten_namespace(Namespace(**value), new_key, separator=separator))
        else:
            items[new_key] = value
    return items


def flatten(d, parent_key='', sep='.'):
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def convert_to_nested_dict(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        keys = key.split('.')
        branch = nested
        for part in keys[:-1]:
            branch = branch.setdefault(part, {})
        branch[keys[-1]] = value == 'True' if isinstance(value, str) and value in ('True', 'False') else value
    return nested


def instantiate_class(input_dict: Dict[str, Any], *args, **kwargs):
    """Copy constructor arguments, apply overrides, then instantiate nested classes."""
    if not isinstance(input_dict, Mapping):
        raise ValueError('[instantiate_class] Expected a class configuration mapping')
    class_path = input_dict.get('class_path')
    if not isinstance(class_path, str) or not class_path or '.' not in class_path:
        raise ValueError('[instantiate_class] class_path must name a module and a class')
    if class_path.startswith('gollum.'):
        class_path = 'alice.' + class_path[len('gollum.'):]
        class_path = class_path.replace('gp_mo_multilayer.DualDeepGP', 'gp_mo_multilayer.MultiDeepGP')
    module_name, class_name = class_path.rsplit('.', 1)
    if not module_name or not class_name:
        raise ValueError('[instantiate_class] class_path must name a module and a class')
    init_args = copy.deepcopy(input_dict.get('init_args', {}))
    if isinstance(init_args, Namespace):
        init_args = vars(init_args)
    if not isinstance(init_args, MutableMapping):
        raise ValueError('[instantiate_class] init_args must be a mapping or Namespace')
    init_args.update(kwargs) # overrides must happen before constructing nested values
    if class_path == 'alice.data.module.BaseDataModule' and init_args.pop('exclude_top', False):
        raise ValueError('[ALICE] exclude_top=True is not supported')
    if class_path == 'alice.data.module.Featurizer':
        init_args.pop('pooling_method', None)
    for name, value in init_args.items():
        if isinstance(value, dict) and 'class_path' in value:
            init_args[name] = instantiate_class(value)
    constructor = getattr(importlib.import_module(module_name), class_name)
    if not callable(constructor):
        raise ValueError(f'[instantiate_class] {class_path} is not callable')
    return constructor(*args, **init_args)


def ctor_supports(param_name: str, cls: Type):
    try:
        return param_name in inspect.signature(cls.__init__).parameters
    except Exception:
        try:
            return param_name in tuple(cls.__init__.__code__.co_varnames)
        except Exception:
            return False


def integer_input(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        limit = f' through {maximum}' if maximum is not None else ' or greater'
        raise ValueError(f'[inputs] {name} must be an integer from {minimum}{limit}')
    return int(value)


def positive_input(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'[inputs] {name} must be finite and positive')
    return value


def validate_class_spec(value, name, allow_short_name=False, path_validator=None):
    if not isinstance(value, dict):
        raise ValueError(f'[recipe] {name} must be a class configuration mapping')
    path = value.get('class_path')
    if (not isinstance(path, str) or ('.' not in path and not allow_short_name)
            or not all(part.isidentifier() for part in path.split('.'))):
        raise ValueError(f'[recipe] {name}.class_path must be a dotted Python class name')
    if path_validator is not None:
        path_validator(path, name) # callers can restrict classes before constructor checks
    if not isinstance(value.get('init_args', {}), dict):
        raise ValueError(f'[recipe] {name}.init_args must be a mapping')
    for key, child in value.get('init_args', {}).items():
        if isinstance(child, dict) and 'class_path' in child:
            validate_class_spec(child, f'{name}.init_args.{key}', path_validator=path_validator)

def validate_bo_data(config):
    """Check the dataset fields and supported batch selection before loading it."""
    data = config['data'].get('init_args', {})
    if config['bo'].get('init_args', {}).get('batch_mode', 'greedy') != 'greedy':
        raise ValueError('ALICE supports greedy batch selection only')
    path = data.get('data_path')
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError('[recipe] data.init_args.data_path must name the input dataset')
    for name in ('normalize_input', 'maximize'):
        if name not in data:
            raise ValueError(f'[recipe] data.init_args.{name} is required')


class BOInputs:
    """Validate BO counts and output paths without touching data or random state."""

    def __init__(self, config):
        self.seed = integer_input(config.get('seed', 1), 'seed', minimum=0, maximum=2**32 - 1)
        self.n_iters = integer_input(config.get('n_iters', 1), 'n_iters')
        self.batch_size = integer_input(config['bo'].get('init_args', {}).get('batch_size', 5), 'batch_size')
        self.reserve_size = integer_input(config.get('reserve_size', 6), 'reserve_size', minimum=0)
        root = config.get('out_root', './results_MOBBOL')
        if not isinstance(root, (str, os.PathLike)) or not str(root).strip():
            raise ValueError('[inputs] out_root must name an output directory')
