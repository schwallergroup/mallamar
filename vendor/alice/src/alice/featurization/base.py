# Dispatch input representations to their featurization functions.
import inspect
from typing import Optional
from alice.featurization.text import get_tokens


class Featurizer:
    def __init__(self, representation: Optional[str] = None, model_name: Optional[str] = None):
        self.representation = representation
        self.params = {'model_name': model_name}
        self._featurization_registry = self._build_registry()

    def _build_registry(self):
        return {'get_tokens': get_tokens}

    def featurize(self, data):
        data_list = data.tolist() if hasattr(data, 'tolist') else data
        if self.representation not in self._featurization_registry:
            raise ValueError(f'Unsupported representation: {self.representation}')
        function = self._featurization_registry[self.representation]
        parameters = inspect.signature(function).parameters
        valid_params = {key: value for key, value in self.params.items() if key in parameters}
        return function(data_list, **valid_params)
