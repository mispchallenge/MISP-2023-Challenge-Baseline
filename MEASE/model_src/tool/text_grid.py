#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import codecs

# class definition
class TextGrid(object):
    def __init__(self, file_type='', object_class='', xmin=0., xmax=0., tiers_status='', tiers=[]):
        self.file_type = file_type
        self.object_class = object_class
        self.xmin = xmin
        self.xmax = xmax
        self.tiers_status = tiers_status
        self.tiers = tiers

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        new_xmax = xend - xstart + self.xmin
        new_xmin = self.xmin
        new_tiers = []

        for tier in self.tiers:
            new_tiers.append(tier.cutoff(xstart=xstart, xend=xend))
        return TextGrid(file_type=self.file_type, object_class=self.object_class, xmin=new_xmin, xmax=new_xmax,
                        tiers_status=self.tiers_status, tiers=new_tiers)


class Tier(object):
    def __init__(self, tier_class='', name='', xmin=0., xmax=0., intervals=[]):
        self.tier_class = tier_class
        self.name = name
        self.xmin = xmin
        self.xmax = xmax
        self.intervals = intervals

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))

    def cutoff(self, xstart=None, xend=None):
        if xstart is None:
            xstart = self.xmin

        if xend is None:
            xend = self.xmax

        if xend < xstart:
            raise ValueError('xend ({}) < xstart ({})'.format(xend, xstart))

        bias = xstart - self.xmin
        new_xmax = xend - bias
        new_xmin = self.xmin
        new_intervals = []
        for interval in self.intervals:
            if interval.xmax <= xstart or interval.xmin >= xend:
                pass
            elif interval.xmin < xstart:
                new_intervals.append(Interval(xmin=new_xmin, xmax=interval.xmax - bias, text=interval.text))
            elif interval.xmax > xend:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=new_xmax, text=interval.text))
            else:
                new_intervals.append(Interval(xmin=interval.xmin - bias, xmax=interval.xmax - bias, text=interval.text))

        return Tier(tier_class=self.tier_class, name=self.name, xmin=new_xmin, xmax=new_xmax, intervals=new_intervals)


class Interval(object):
    def __init__(self, xmin=0., xmax=0., text=''):
        self.xmin = xmin
        self.xmax = xmax
        self.text = text

        if self.xmax < self.xmin:
            raise ValueError('xmax ({}) < xmin ({})'.format(self.xmax, self.xmin))


# io
def read_textgrid_from_file(filepath):
    with codecs.open(filepath, 'r', encoding='utf8') as handle:
        lines = handle.readlines()
    if lines[-1] == '\r\n':
        lines = lines[:-1]

    assert 'File type' in lines[0], 'error line 0, {}'.format(lines[0])
    file_type = lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'Object class' in lines[1], 'error line 1, {}'.format(lines[1])
    object_class = lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert lines[2] == '\r\n', 'error line 2, {}'.format(lines[2])

    assert 'xmin' in lines[3], 'error line 3, {}'.format(lines[3])
    xmin = float(lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in lines[4], 'error line 4, {}'.format(lines[4])
    xmax = float(lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'tiers?' in lines[5], 'error line 5, {}'.format(lines[5])
    tiers_status = lines[5].split('?')[1].replace(' ', '').replace('\r', '').replace('\n', '')

    assert 'size' in lines[6], 'error line 6, {}'.format(lines[6])
    size = int(lines[6].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert lines[7] == 'item []:\r\n', 'error line 7, {}'.format(lines[7])

    tier_start = []
    for item_idx in range(size):
        tier_start.append(lines.index(' ' * 4 + 'item [{}]:\r\n'.format(item_idx + 1)))

    tier_end = [*tier_start[1:], len(lines)]

    tiers = []
    for tier_idx in range(size):
        tiers.append(read_tier_from_lines(tier_lines=lines[tier_start[tier_idx] + 1: tier_end[tier_idx]]))

    return TextGrid(file_type=file_type, object_class=object_class, xmin=xmin, xmax=xmax, tiers_status=tiers_status,
                    tiers=tiers)


def write_textgrid_to_file(filepath, textgrid):
    lines = [
        'File type = "{}"\r\n'.format(textgrid.file_type),
        'Object class = "{}"\r\n'.format(textgrid.object_class),
        '\r\n',
        'xmin = {}\r\n'.format(textgrid.xmin),
        'xmax = {}\r\n'.format(textgrid.xmax),
        'tiers? {}\r\n'.format(textgrid.tiers_status),
        'size =  {}\r\n'.format(len(textgrid.tiers)),
        'item []:\r\n'
    ]
    for tier_idx, tier in enumerate(textgrid.tiers):
        lines.append(' ' * 4 + 'item [{}]:\r\n'.format(tier_idx + 1))
        lines.extend(write_tier_to_lines(tier=tier))

    lines.append('\r\n')

    with codecs.open(filepath, 'w', encoding='utf8') as handle:
        handle.write(''.join(lines))
    return None


def read_tier_from_lines(tier_lines):
    assert 'class' in tier_lines[0], 'error line 0, {}'.format(tier_lines[0])
    tier_class = tier_lines[0].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'name' in tier_lines[1], 'error line 1, {}'.format(tier_lines[1])
    name = tier_lines[1].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    assert 'xmin' in tier_lines[2], 'error line 2, {}'.format(tier_lines[2])
    xmin = float(tier_lines[2].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in tier_lines[3], 'error line 3, {}'.format(tier_lines[3])
    xmax = float(tier_lines[3].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'intervals: size' in tier_lines[4], 'error line 4, {}'.format(tier_lines[4])
    intervals_num = int(tier_lines[4].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert len(tier_lines[5:]) == intervals_num * 5, 'error lines'

    intervals = []
    for intervals_idx in range(intervals_num):
        assert tier_lines[5 + 5 * intervals_idx + 0] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
        assert tier_lines[5 + 5 * intervals_idx + 1] == ' ' * 8 + 'intervals [{}]:\r\n'.format(intervals_idx + 1)
        intervals.append(read_interval_from_lines(
            interval_lines=tier_lines[7 + 5 * intervals_idx: 10 + 5 * intervals_idx]))
    return Tier(tier_class=tier_class, name=name, xmin=xmin, xmax=xmax, intervals=intervals)


def write_tier_to_lines(tier):
    tier_lines = [
        ' ' * 8 + 'class = "{}"\r\n'.format(tier.tier_class),
        ' ' * 8 + 'name = "{}"\r\n'.format(tier.name),
        ' ' * 8 + 'xmin = {}\r\n'.format(tier.xmin),
        ' ' * 8 + 'xmax = {}\r\n'.format(tier.xmax),
        ' ' * 8 + 'intervals: size = {}\r\n'.format(len(tier.intervals)),
    ]

    for interval_idx, interval in enumerate(tier.intervals):
        tier_lines.append(' ' * 8 + 'intervals [{}]:\r\n'.format(interval_idx + 1))
        tier_lines.append(' ' * 8 + 'intervals [{}]:\r\n'.format(interval_idx + 1))
        tier_lines.extend(write_interval_to_lines(interval=interval))
    return tier_lines


def read_interval_from_lines(interval_lines):
    assert len(interval_lines) == 3, 'error lines'

    assert 'xmin' in interval_lines[0], 'error line 0, {}'.format(interval_lines[0])
    xmin = float(interval_lines[0].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'xmax' in interval_lines[1], 'error line 1, {}'.format(interval_lines[1])
    xmax = float(interval_lines[1].split('=')[1].replace(' ', '').replace('\r', '').replace('\n', ''))

    assert 'text' in interval_lines[2], 'error line 2, {}'.format(interval_lines[2])
    text = interval_lines[2].split('=')[1].replace(' ', '').replace('"', '').replace('\r', '').replace('\n', '')

    return Interval(xmin=xmin, xmax=xmax, text=text)


def write_interval_to_lines(interval):
    interval_lines = [
        ' ' * 12 + 'xmin = {}\r\n'.format(interval.xmin),
        ' ' * 12 + 'xmax = {}\r\n'.format(interval.xmax),
        ' ' * 12 + 'text = "{}"\r\n'.format(interval.text),
    ]
    return interval_lines


if __name__ == '__main__':
    checkout_tg = read_textgrid_from_file(filepath='D:\\Code\\python_project\\Embedding_Aware_Speech_Enhancement_edition_3\\Textgrid_C0001\\1.TextGrid')
    cut_tg = checkout_tg.cutoff(xstart=220)
    write_textgrid_to_file(filepath='D:\\Code\\python_project\\Embedding_Aware_Speech_Enhancement_edition_3\\Textgrid_C0001\\1_cut_220.TextGrid', textgrid=cut_tg)
