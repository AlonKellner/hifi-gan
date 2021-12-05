from pathlib import Path

from textgrid import TextGrid, IntervalTier
import pandas as pd

from complex_data_parser import get_path_by_glob


def parse_textgrid(subdir, textgrid_pattern):
    textgrid_path = get_path_by_glob(subdir, textgrid_pattern)
    path_str = str(textgrid_path)
    textgrid = TextGrid.fromFile(path_str)
    return {
        tier.name: get_annotations_dataframe(tier) for tier in textgrid if isinstance(tier, IntervalTier)
    }


def get_annotations_dataframe(tier):
    rows = [{
                'start': interval.minTime,
                'end': interval.maxTime,
                'text': interval.mark
            } for interval in tier.intervals]
    result_data_frame = pd.DataFrame(rows)
    return result_data_frame


def main():
    subdir = Path('/datasets/LibriSpeech')
    textgrid_pattern = '**/librispeech_alignments/test-clean/6930/76324/6930-76324-0017.TextGrid'
    result = parse_textgrid(subdir, textgrid_pattern)
    print(result)


if __name__ == '__main__':
    main()
