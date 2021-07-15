LRELU_SLOPE = 0.1


def get_static_periods_config():
    return [1, 2, 3, 5, 8, 13]


def get_static_generator_config():
    generator_config = (
        (
            [
                ("conv", (1, 16, 31, 1)),
                ("res_fusion", (16, get_res_block_config(31))),
                ("lrelu", LRELU_SLOPE),
                ("conv", (16, 64, 7, 4)),
                ("res_fusion", (64, get_res_block_config(3))),
                ("lrelu", LRELU_SLOPE)
            ],
            [[('conv', (64, 32, 7, 1))], [('conv', (64, 32, 7, 1))]]
        ),
        (
            [[('conv', (32, 64, 3, 1))], [('conv', (32, 64, 3, 1))]],
            [
                ("res_fusion", (64, get_res_block_config(3))),
                ("lrelu", LRELU_SLOPE),
                ("trans", (64, 16, 7, 4)),
                ("res_fusion", (16, get_res_block_config(31))),
                ("lrelu", LRELU_SLOPE),
                ("trans", (16, 1, 31, 1)),
                ("tanh",)
            ]
        )
    )
    return generator_config


def get_res_block_config(kernel_size):
    common_res_blocks = \
        [
            [
                [(kernel_size, 1), (kernel_size, 1)],
                [(kernel_size, 2), (kernel_size, 1)]
            ],
            [
                [(kernel_size, 2), (kernel_size, 1)],
                [(kernel_size, 6), (kernel_size, 1)]
            ],
            [
                [(kernel_size, 3),  (kernel_size, 1)],
                [(kernel_size, 12), (kernel_size, 1)]
            ]
        ]
    return common_res_blocks
