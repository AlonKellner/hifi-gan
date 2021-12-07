def rank(prefix: str):
    elements = prefix.split('/')
    elements[0] = f'{elements[0]}_{len(elements)}'
    return '/'.join(elements)

