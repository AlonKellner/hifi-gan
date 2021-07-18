
def get_padding(kernel_size, stride=1, dilation=1):
    return ((kernel_size - 1)*dilation + 1)//2


def get_padding_trans(kernel_size, stride=1, dilation=1):
    total_padding = (kernel_size - 1)*dilation + 1 - stride
    return total_padding//2 + total_padding % 2, total_padding % 2
