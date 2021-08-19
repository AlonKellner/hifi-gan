from tgt import TextGrid


def parse_timed_tags_from_textgrid(textgrid_path, interval):
    textgrid = TextGrid(textgrid_path)
    tiers = textgrid.tiers
    start = textgrid.start_time
    end = textgrid.end_time
    return {
        tier.name: tier.get_annotations_by_time(current_time)
        for tier, current_time in zip(tiers, range(start, end, interval))
    }
