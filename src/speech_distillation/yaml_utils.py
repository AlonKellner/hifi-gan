import yaml
from pathlib import Path

from yaml import FullLoader


def do_and_cache_dict(func, cache_format):
    result = func()
    output = {}
    for key, value in result.items():
        cache_path = cache_format.format(key)
        if Path(cache_path).exists():
            with open(cache_path, 'r') as cache:
                output[key] = yaml.load(cache, FullLoader)
        else:
            with open(cache_path, 'w') as cache:
                output[key] = value
                yaml.dump(value, cache)

    return output


def do_and_cache(func, cache_path, copy_path=None):
    read_path = cache_path if copy_path is None else copy_path
    write_path = cache_path
    if Path(read_path).exists():
        with open(read_path, 'r') as cache:
            result = yaml.load(cache, FullLoader)
    else:
        result = func()

    with open(write_path, 'w') as cache:
        yaml.dump(result, cache)

    return result
