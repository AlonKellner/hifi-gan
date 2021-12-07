def get_recursive(func, *data, args=[], kwargs={}, **kwdata):
    if isinstance(data[0], dict):
        return {get_recursive(func, *(d[key] for d in data), **{n: v[key] for n, v in kwdata}, args=args, kwargs=kwargs) for key in data[0]}
    elif isinstance(data[0], list):
        return [do_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata}, args=args, kwargs=kwargs) for index in range(len(data))]
    elif isinstance(data[0], tuple):
        return (do_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata}, args=args, kwargs=kwargs) for index in range(len(data)))
    else:
        return func(*data, *args, **kwargs, **kwdata)


def do_recursive(func, *data, args=[], kwargs={}, **kwdata):
    if isinstance(data[0], dict):
        for key in data[0]:
            do_recursive(func, *(d[key] for d in data), **{n: v[key] for n, v in kwdata}, args=args, kwargs=kwargs)
    elif isinstance(data[0], (list, tuple)):
        for index in range(len(data)):
            do_recursive(func, *(d[index] for d in data), **{n: v[index] for n, v in kwdata}, args=args, kwargs=kwargs)
    else:
        func(*data, *args, **kwargs, **kwdata)
