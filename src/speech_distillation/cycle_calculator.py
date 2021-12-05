import itertools
import math


def cycles_valid(cycles, a, b):
    if not sum(cycles) == b:
        return False
    for index, cycle in enumerate(cycles):
        index = index + 1
        if not cycle >= 3:
            return False
        if not cycle > index:
            return False
        if not cycle <= a:
            return False
        if not(index == 1 or math.gcd(cycle, index) == 1):
            return False
    return True


def get_remainder_from_permutation(permutation):
    remainder = []
    index = 0
    for b in permutation:
        if b:
            index += 1
        else:
            remainder.append(index)
            index = 0
    remainder.append(index)
    remainder.sort()
    return tuple(remainder)


def get_all_remainders(remainder, buckets, max_value):
    if remainder == 0:
        return {(0, )}
    items = [*([True]*remainder), *([False]*(buckets-1))]
    permutations = set(itertools.permutations(items))
    all_remainders = set(get_remainder_from_permutation(permutation) for permutation in permutations)
    all_remainders = set(remainder_set for remainder_set in all_remainders if max(remainder_set) <= max_value and min(remainder_set) > 0)
    return all_remainders


def calculate_cycles(a, b):
    assert a >= 3
    assert a*a - a >= b

    cycles = []
    count = 0
    while not cycles_valid(cycles, a, b) and count <= b / 3:
        count += 1
        cycles.insert(0, min(a, b))

        if b <= sum(cycles):
            remainder = sum(cycles) - b
            buckets = 0
            while buckets < len(cycles):
                buckets += 1
                all_remainders = get_all_remainders(remainder, buckets, a-3)
                for remainder_tuple in all_remainders:
                    cycles_copy = cycles.copy()
                    for index, item in enumerate(remainder_tuple):
                        cycles_copy[-(index+1)] -= item
                    cycles_copy.sort()
                    permutations = set(itertools.permutations(cycles_copy))
                    valid_permutations = [permutation for permutation in permutations if cycles_valid(permutation, a, b)]
                    valid_permutations.sort(key=lambda p: len(p))
                    if len(valid_permutations) > 0:
                        cycles_copy = tuple(cycles_copy)
                        if cycles_copy in valid_permutations:
                            return cycles_copy
                        else:
                            return valid_permutations[0]
    raise AssertionError('Cycle sequence could not be calculated!')


def assert_fail(func):
    try:
        func()
        raise AssertionError('The call did not fail!')
    except:
        pass


assert_fail(lambda: calculate_cycles(0, 0))
assert calculate_cycles(3, 3) == (3,)
assert calculate_cycles(3, 6) == (3, 3)
assert_fail(lambda: calculate_cycles(3, 7))
assert calculate_cycles(5, 8) == (3, 5)
assert calculate_cycles(5, 13) == (3, 5, 5)
assert calculate_cycles(5, 11) == (3, 3, 5)
assert_fail(lambda: calculate_cycles(4, 8))
assert calculate_cycles(4, 3) == (3,)
assert calculate_cycles(4, 11) == (4, 3, 4)
assert calculate_cycles(5, 14) == (4, 5, 5)
assert calculate_cycles(6, 25) == (6, 3, 5, 5, 6)
assert_fail(lambda: calculate_cycles(3, 12))
assert_fail(lambda: calculate_cycles(7, 50))
assert_fail(lambda: calculate_cycles(5, 21))

for i in range(5, 10):
    for j in range(i, i*i//2):
        print(i, j, calculate_cycles(i, j))
