import pandas as pd
import re
import json
from pathlib import Path


def parse_complex_data(subdir, data_config, result_group):
    all_labels = {}
    for data_group in data_config:
        group_name = data_group['group-name']
        group_labels = parse_group_sources_labels(subdir, data_group)
        all_labels[group_name] = group_labels
        if 'enrichments' in data_group:
            enrichments = data_group['enrichments']
            for enrichment in enrichments:
                parse_enrichment_labels(subdir, group_name, enrichment, all_labels)
    return all_labels[result_group]


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
    elif enrichment_type == 'retype':
        parse_retype_labels(group_name, enrichment, all_labels)
    else:
        raise Exception('Unknown enrichment type - {}'.format(enrichment_type))

    all_labels[group_name] = all_labels[group_name].infer_objects()


def parse_regex_labels(group_name, enrichment, all_labels):
    group_labels = all_labels[group_name]
    patterns = enrichment['patterns']
    all_regex_labels = []
    for label, pattern in patterns.items():
        regex_groups = []
        group_labels.apply(
            axis=1,
            func=lambda row: regex_groups.append(get_regex_of_value(row, label, pattern))
        )
        regex_labels = pd.DataFrame(regex_groups)
        all_regex_labels.append(regex_labels)

    group_labels = all_labels[group_name]
    group_labels = pd.concat([group_labels, *all_regex_labels], axis=1)
    all_labels[group_name] = group_labels


def get_regex_of_value(row, label, pattern):
    match = re.search(resolve_label_references(pattern, row.to_dict()), str(row[label]))
    groups = match.groupdict()
    return groups


def parse_files_labels(subdir, group_name, enrichment, all_labels):
    labels = enrichment['labels']
    group_labels = all_labels[group_name]
    sample_row = group_labels.iloc[[0]].squeeze()
    for label, pattern in labels.items():
        get_path_by_glob(subdir, resolve_label_references(pattern, sample_row.to_dict()))
        group_labels[label] = group_labels.apply(
            axis=1,
            func=lambda row: resolve_label_references(pattern, row.to_dict())
        )

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


def parse_retype_labels(group_name, enrichment, all_labels):
    mapping = enrichment['mapping']

    group_labels = all_labels[group_name]
    for label, new_type in mapping.items():
        if new_type == 'int':
            group_labels[label] = group_labels[label].astype(int)
        if new_type == 'str':
            group_labels[label] = group_labels[label].astype(str)
        if new_type == 'float':
            group_labels[label] = group_labels[label].astype(float)
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
    group_labels = None
    sources = data_group['sources']
    for source in sources:
        source_rows = parse_rows_from_source(subdir, source)
        if group_labels is None:
            group_labels = source_rows
        else:
            group_labels.append(source_rows)
    if 'group-labels' in data_group:
        constant_group_labels = data_group['group-labels']
        for label, constant_value in constant_group_labels.items():
            group_labels[label] = constant_value
    group_labels['subdir'] = subdir
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
        skiprows=source['skiprows'],
        skipinitialspace=True,
        index_col=False
    )
    csv.columns = map(str.strip, csv.columns)
    return csv


def parse_rows_from_glob(subdir, source):
    source_glob = source['glob']
    glob_files = list(subdir.glob(source_glob))

    label = source['label']
    return pd.DataFrame(glob_files, columns=[label])


def get_path_by_glob(subdir, glob_pattern):
    subdir = Path(subdir)
    files_list = [path for path in subdir.glob(glob_pattern)]
    if len(files_list) == 0:
        raise Exception('Missing file [{}] in [{}]'.format(glob_pattern, str(subdir)))
    file_path = min(files_list, key=lambda x: len(str(x)))
    return file_path


def resolve_label_references(input_to_resolve: str, labels):
    for label, value in labels.items():
        input_to_resolve = input_to_resolve.replace('{{{}}}'.format(label), str(value))
    return input_to_resolve


def main():
    data_config_path = 'data_config/data_config.json'

    with open(data_config_path) as f:
        data = f.read()
    data_config = json.loads(data)
    result = parse_complex_data(subdir=Path('/datasets/LibriSpeech'), data_config=data_config)
    print(result['wavs']['textgrid'].iloc[[0]].squeeze())


if __name__ == '__main__':
    main()
