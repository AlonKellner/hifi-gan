import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(torch.nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models

    def forward(self, *x):
        results = [model(*x) for model in self.models]
        stacked = self.get_recursive_array_function(results, lambda res: torch.stack(res, dim=0))
        means = self.get_recursive_function(stacked, lambda res: res.mean(dim=0))
        variances = self.get_recursive_function(stacked, lambda res: res.var(dim=0))
        return {'mean': means, 'variance': variances}

    def get_recursive_array_function(self, results, function):
        if isinstance(results[0], list):
            return [
                self.get_recursive_array_function([result[i] for result in results], function)
                for i in range(len(results[0]))
            ]
        elif isinstance(results[0], tuple):
            return tuple(
                self.get_recursive_array_function([result[i] for result in results], function)
                for i in range(len(results[0]))
            )
        elif isinstance(results[0], dict):
            return {
                key: self.get_recursive_array_function([result[key] for result in results], function)
                for key in results[0].keys()
            }
        else:
            return function(results)

    def get_recursive_function(self, result, function):
        if isinstance(result, list):
            return [
                self.get_recursive_function(sub_result, function) for sub_result in result
            ]
        elif isinstance(result, tuple):
            return tuple(
                self.get_recursive_function(sub_result, function) for sub_result in result
            )
        elif isinstance(result, dict):
            return {
                key: self.get_recursive_function(sub_result, function) for key, sub_result in result.items()
            }
        else:
            return function(result)
