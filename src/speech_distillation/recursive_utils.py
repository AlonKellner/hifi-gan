def get_recursive(func, *data, args=[], kwargs={}, **kwdata):
    example = next(iter(data)) if len(data) > 0 else next(iter(kwdata.values()))
    if isinstance(example, dict):
        return {key: get_recursive(func, *(d[key] for d in data), **{n: v[key] for n, v in kwdata.items()}, args=args, kwargs=kwargs) for key in example}
    elif isinstance(example, list):
        return [get_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata.items()}, args=args, kwargs=kwargs) for index in range(len(example))]
    elif isinstance(example, tuple):
        return (get_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata.items()}, args=args, kwargs=kwargs) for index in range(len(example)))
    else:
        return func(*data, *args, **kwargs, **kwdata)


def do_recursive(func, *data, args=[], kwargs={}, **kwdata):
    example = next(iter(data)) if len(data) > 0 else next(iter(kwdata.values()))
    if isinstance(example, dict):
        for key in example:
            do_recursive(func, *(d[key] for d in data), **{n: v[key] for n, v in kwdata.items()}, args=args, kwargs=kwargs)
    elif isinstance(example, (list, tuple)):
        for index in range(len(example)):
            do_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata.items()}, args=args, kwargs=kwargs)
    else:
        func(*data, *args, **kwargs, **kwdata)
