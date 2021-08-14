import pandas as pd
import re


def parse_complex_data(subdir, data_config):
    all_labels = {}
    for data_group in data_config:
        group_name = data_group['group-name']
        group_labels = parse_group_sources_labels(subdir, data_group)
        if 'enrichments' in data_group:
            enrichments = data_group['enrichments']
            for enrichment in enrichments:
                parse_enrichment_labels(subdir, group_name, enrichment, all_labels)

        all_labels[group_name] = group_labels


def parse_enrichment_labels(subdir, group_name, enrichment, all_labels):
    enrichment_type = enrichment['type']
    if enrichment_type == 'regex':
        parse_regex_labels(group_name, enrichment, all_labels)
    elif enrichment_type == 'files':
        parse_files_labels(subdir, group_name, enrichment, all_labels)
    elif enrichment_type == 'join':
        parse_join_labels(group_name, enrichment, all_labels)
    elif enrichment_type == 'select':
        parse_select_labels(group_name, enrichment, all_labels)
    elif enrichment_type == 'rename':
        parse_rename_labels(group_name, enrichment, all_labels)
    else:
        raise Exception('Unknown enrichment type - {}'.format(enrichment_type))


def parse_regex_labels(group_name, enrichment, all_labels):
    group_labels = all_labels[group_name]
    patterns = enrichment['patterns']
    all_regex_labels = []
    for label, pattern in patterns.items():
        regex_groups = []
        group_labels.apply(
            lambda row: regex_groups.append(
                re.search(row[label], resolve_label_references(pattern, row.to_dict())).groups()
            )
        )
        regex_labels = pd.DataFrame(regex_groups)
        all_regex_labels.append(regex_labels)

    group_labels = all_labels[group_name]
    group_labels = pd.concat([group_labels, *all_regex_labels], axis=1)
    all_labels[group_name] = group_labels


def parse_files_labels(subdir, group_name, enrichment, all_labels):
    labels = enrichment['labels']
    group_labels = all_labels[group_name]

    for label, pattern in labels.items():
        group_labels[label] = group_labels.apply(
            lambda row: get_path_by_glob(subdir, resolve_label_references(pattern, row.to_dict()))
        )

    group_labels = group_labels[labels]
    all_labels[group_name] = group_labels


def parse_select_labels(group_name, enrichment, all_labels):
    labels = enrichment['labels']

    group_labels = all_labels[group_name]
    group_labels = group_labels[labels]
    all_labels[group_name] = group_labels


def parse_rename_labels(group_name, enrichment, all_labels):
    mapping = enrichment['mapping']

    group_labels = all_labels[group_name]
    group_labels = group_labels.rename(columns=mapping)
    all_labels[group_name] = group_labels


def parse_join_labels(group_name, enrichment, all_labels):
    other_group_name = enrichment['other-group-name']
    base_label = enrichment['base-label']
    other_label = enrichment['other-label']

    group_labels = all_labels[group_name]
    other_group_label = all_labels[other_group_name]
    group_labels = pd.merge(group_labels, other_group_label, left_on=base_label, right_on=other_label)
    all_labels[group_name] = group_labels


def parse_group_sources_labels(subdir, data_group):
    group_labels = pd.DataFrame()
    sources = data_group['sources']
    for source in sources:
        group_labels.append(parse_rows_from_source(subdir, source))
    if 'group-labels' in data_group:
        constant_group_labels = data_group['group-labels']
        for label, constant_value in constant_group_labels.items():
            group_labels[label] = constant_value
    return group_labels


def parse_rows_from_source(subdir, source):
    source_type = source['type']
    if source_type == 'csv':
        return parse_rows_from_csv(subdir, source)
    elif source_type == 'glob':
        return parse_rows_from_glob(subdir, source)
    else:
        raise Exception('Unknown source type - {}'.format(source_type))


def parse_rows_from_csv(subdir, source):
    path = get_path_by_glob(subdir, source['path'])
    csv = pd.read_csv(
        path,
        delimiter=source['delimiter'],
        skiprows=source['skiprows']
    )
    return csv


def parse_rows_from_glob(subdir, source):
    source_glob = source['glob']
    glob_files = list(subdir.glob(source_glob))

    label = source['label']
    return pd.DataFrame(glob_files, columns=[label])


def get_path_by_glob(subdir, path):
    return min(subdir.glob(path), key=lambda x: len(str(x)))


def resolve_label_references(input_to_resolve: str, labels):
    for label, value in labels.items():
        input_to_resolve = input_to_resolve.replace('{{{}}}'.format(label), value)
    return input_to_resolve
