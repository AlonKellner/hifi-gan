import math


def get_padding(kernel, stride=(1, 1), dilation=(1, 1)):
    if isinstance(kernel, int):
        return get_1d_padding(kernel, stride, dilation)
    if isinstance(kernel, tuple):
        return tuple(get_1d_padding(*conv_params) for conv_params in zip(kernel, stride, dilation))


def get_1d_padding(kernel, stride=1, dilation=1):
    return ((kernel - 1)*dilation + 1)//2


def get_padding_trans(kernel, stride, dilation):
    if isinstance(kernel, int):
        return get_1d_padding_trans(kernel, stride, dilation)
    if isinstance(kernel, tuple):
        return tuple(get_1d_padding_trans(*conv_params) for conv_params in zip(kernel, stride, dilation))


def get_1d_padding_trans(kernel, stride=1, dilation=1):
    total_padding = (kernel - 1) * dilation + 1 - stride
    return total_padding//2 + total_padding % 2, total_padding % 2


def get_padding_period(length, period):
    if isinstance(length, int):
        return get_1d_padding_period(length, period)
    if isinstance(length, tuple):
        return tuple(get_1d_padding_period(*conv_params) for conv_params in zip(length, period))


def get_1d_padding_period(length, periods_count):
    total_padding = int(math.ceil(length/periods_count)*periods_count-length)
    return total_padding//2, (total_padding//2) + (total_padding % 2)
