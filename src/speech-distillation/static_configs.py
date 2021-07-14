LRELU_SLOPE = 0.1


def get_static_generator_config():
    generator_config = (
        (
            get_static_vo_encoder_config(),
            [[('conv', (128, 64, 3, 1))], [('conv', (128, 64, 3, 1))]]
        ),
        (
            [[('conv', (64, 128, 1, 1))], [('conv', (64, 128, 1, 1))]],
            get_static_vo_decoder_config()
        )
    )
    return generator_config


def get_static_vo_encoder_config():
    encoder_config = [
        ("conv", (1, 16, 15, 4)),
        ("res_fusion", (16, get_res_block_config(15))),
        ("lrelu", LRELU_SLOPE),
        ("conv", (16, 64, 7, 4)),
        ("res_fusion", (64, get_res_block_config(7))),
        ("lrelu", LRELU_SLOPE),
        ("conv", (64, 128, 3, 2)),
        ("res_fusion", (128, get_res_block_config(3))),
        ("lrelu", LRELU_SLOPE)
    ]
    return encoder_config


def get_static_vo_decoder_config():
    decoder_config = [
        ("res_fusion", (128, get_res_block_config(3))),
        ("lrelu", LRELU_SLOPE),
        ("trans", (128, 64, 4, 2)),
        ("res_fusion", (64, get_res_block_config(7))),
        ("lrelu", LRELU_SLOPE),
        ("trans", (64, 16, 8, 4)),
        ("res_fusion", (16, get_res_block_config(15))),
        ("lrelu", LRELU_SLOPE),
        ("trans", (16, 1, 16, 4)),
        ("tanh",)
    ]
    return decoder_config


def get_res_block_config(kernel_size):
    common_res_blocks = \
        [
            [
                [(kernel_size, 1), (kernel_size, 1)],
                [(kernel_size, 2), (kernel_size, 1)],
                [(kernel_size, 3), (kernel_size, 1)]
            ],
            [
                [(kernel_size, 3), (kernel_size, 1)],
                [(kernel_size, 5), (kernel_size, 1)],
                [(kernel_size, 8), (kernel_size, 1)]
            ],
            [
                [(kernel_size, 8),  (kernel_size, 1)],
                [(kernel_size, 13), (kernel_size, 1)],
                [(kernel_size, 21), (kernel_size, 1)]
            ]
        ]
    return common_res_blocks
