import yaml
from pathlib import Path

from yaml import FullLoader


def do_and_cache(func, cache_path):
    if Path(cache_path).exists():
        with open(cache_path, 'r') as cache:
            result = yaml.load(cache, FullLoader)
    else:
        result = func()
        with open(cache_path, 'w') as cache:
            yaml.dump(result, cache)

    return result
