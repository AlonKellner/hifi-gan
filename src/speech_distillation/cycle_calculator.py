import itertools
import math


def cycles_valid(cycles, a, b, min_cycle):
    if not sum(cycles) == b:
        return False
    for index, cycle in enumerate(cycles):
        index = index + 1
        if cycle < min_cycle:
            return False
        if cycle <= index:
            return False
        if cycle > a:
            return False
        if cycle % index == 0 and cycle // index < min_cycle:
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


def get_superpermutations(remainder, buckets):
    if buckets == 1:
        return [[True]*remainder]
    superpermutations = []
    for cutoff in range(remainder+1):
        sub_superpermutations = get_superpermutations(remainder-cutoff, buckets-1)
        for sub_superpermutation in sub_superpermutations:
            superpermutations.append([*([True]*cutoff), False, *sub_superpermutation])
    return superpermutations


def get_all_remainders(remainder, buckets, max_value):
    if remainder == 0:
        return [(0, )]
    permutations = get_superpermutations(remainder, buckets)
    permutations = set(tuple(permutation) for permutation in permutations)
    all_remainders = set(get_remainder_from_permutation(permutation) for permutation in permutations)
    all_remainders = set(remainder_set for remainder_set in all_remainders if max(remainder_set) <= max_value and min(remainder_set) > 0)
    return list(all_remainders)


def calculate_cycles(a, b, min_cycle=3):
    assert a >= min_cycle
    common_divisors = [a % cycle == 0 for cycle in range(2, min_cycle)].count(True)
    assert a*a - a - common_divisors >= b

    cycles = []
    count = 0
    while not cycles_valid(cycles, a, b, min_cycle) and count <= b / min_cycle and count < a:
        count += 1
        cycles.insert(0, min(a, b))

        if b <= sum(cycles):
            remainder = sum(cycles) - b
            buckets = 0
            while buckets < len(cycles):
                buckets += 1
                all_remainders = get_all_remainders(remainder, buckets, a-3)
                all_remainders.sort(key=lambda r: min(r), reverse=True)
                for remainder_tuple in all_remainders:
                    cycles_copy = cycles.copy()
                    for index, item in enumerate(remainder_tuple):
                        cycles_copy[-(index+1)] -= item
                    cycles_copy.sort()
                    cycles_copy = tuple(cycles_copy)
                    if cycles_valid(cycles_copy, a, b, min_cycle):
                        return cycles_copy
                    permutations = set(itertools.permutations(cycles_copy))
                    valid_permutations = [permutation for permutation in permutations if cycles_valid(permutation, a, b, min_cycle)]
                    valid_permutations.sort(key=lambda p: sum(v*ind for ind, v in enumerate(p)), reverse=True)
                    if len(valid_permutations) > 0:
                        return valid_permutations[0]
    raise AssertionError('Cycle sequence could not be calculated!')


def assert_fail(func):
    try:
        func()
        raise AssertionError('The call did not fail!')
    except:
        pass


if __name__ == '__main__':
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
    assert calculate_cycles(6, 25) == (3, 6, 4, 6, 6)
    assert_fail(lambda: calculate_cycles(3, 12))
    assert_fail(lambda: calculate_cycles(7, 50))
    assert_fail(lambda: calculate_cycles(5, 21))
    assert calculate_cycles(6, 28) == (6, 6, 4, 6, 6)
    assert calculate_cycles(10, 39) == (9, 10, 10, 10)
    assert calculate_cycles(10, 49) == (10, 10, 10, 10, 9)
    assert calculate_cycles(10, 50) == (4, 10, 10, 10, 6, 10)
    assert calculate_cycles(12, 12) == (12,)
    assert calculate_cycles(5, 20) == (5, 5, 5, 5)

    min_cycle = 3
    for i in range(5, 13):
        common_divisors = [i % cycle == 0 for cycle in range(2, min_cycle)].count(True)
        for j in range(i, i*i - i - common_divisors + 1):
            cycles = calculate_cycles(i, j, min_cycle)
            print(i, j, cycles, len(cycles), cycles.count(i))
