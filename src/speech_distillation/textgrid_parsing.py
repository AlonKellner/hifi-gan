from datetime import datetime, timedelta

import pandas as pd
from tgt.io import read_textgrid
import tgt
from pathlib import Path
import numpy as np
from complex_data_parser import get_path_by_glob


def parse_textgrid(subdir, textgrid_pattern):
    textgrid_path = get_path_by_glob(subdir, textgrid_pattern)
    textgrid = read_textgrid(str(textgrid_path))
    tiers = textgrid.tiers
    return {
        tier.name: get_annotations_dataframe(tier) for tier in tiers
    }


def get_annotations_dataframe(tier):
    annotations = tier.annotations
    rows = [{
                'start': annotation.start_time,
                'end': annotation.end_time,
                'text': annotation.text
            } for annotation in annotations]
    result_data_frame = pd.DataFrame(rows)
    return result_data_frame


def main():
    subdir = Path('/datasets/LibriSpeech')
    textgrid_pattern = '**/librispeech_alignments/test-clean/6930/76324/6930-76324-0017.TextGrid'
    result = parse_textgrid(subdir, textgrid_pattern)
    print(result)


if __name__ == '__main__':
    main()
