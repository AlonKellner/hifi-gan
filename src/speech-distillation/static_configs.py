LRELU_SLOPE = 0.1


def get_static_all_in_one_discriminator():
    return [
        ([('conv_rech', (1, 1, 33, 1, 1, 1, 'spectral'))]),
    ]


def get_all_in_one_block_config(scale):
    return (
        (),
        [
            ('pool', (2 * scale - 1, scale)),
            ('period', (scale,))
        ],
        ('res', ())
    )


def get_static_scale_discriminators_config():
    spectral_discriminator_config = [
        (('conv', (1, 32, 31, 4, 1, 1, 'spectral')), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (32, 128, 15, 4, 1, 2, 'spectral')), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (128, 512, 7, 4, 1, 4, 'spectral')), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (512, 1024, 3, 4, 1, 16, 'spectral')), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (1024, 1, 15, 1, 1, 1, 'spectral')), True),
    ]
    default_discriminator_config = [
        (('conv', (1, 32, 31, 4, 1, 1)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (32, 128, 15, 4, 1, 2)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (128, 512, 7, 4, 1, 4)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (512, 1024, 3, 4, 1, 16)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv', (1024, 1, 15, 1, 1, 1)), True),
    ]

    return [
        spectral_discriminator_config,
        [(('pool', (3, 2)), False), *default_discriminator_config],
        [(('pool', (5, 3)), False), *default_discriminator_config],
        [(('pool', (9, 5)), False), *default_discriminator_config],
        [(('pool', (15, 8)), False), *default_discriminator_config],
        [(('pool', (25, 13)), False), *default_discriminator_config],
        [(('pool', (41, 21)), False), *default_discriminator_config],
        [(('pool', (67, 34)), False), *default_discriminator_config],
        [(('pool', (109, 55)), False), *default_discriminator_config],
        [(('pool', (177, 89)), False), *default_discriminator_config]
    ]


def get_static_period_discriminators_config():
    default_discriminator_config = [
        (('conv2', (1, 32, (31, 1), (4, 1), (1, 1), 1)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv2', (32, 128, (15, 1), (4, 1), (1, 1), 2)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv2', (128, 512, (7, 1), (2, 1), (1, 1), 4)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv2', (512, 1024, (3, 1), (2, 1), (1, 1), 16)), False),
        (('lrelu', LRELU_SLOPE), True),
        (('conv2', (1024, 1, (15, 1), (1, 1), (1, 1))), True)
    ]

    return [
        (2, default_discriminator_config),
        (3, default_discriminator_config),
        (5, default_discriminator_config),
        (8, default_discriminator_config),
        (13, default_discriminator_config)
    ]


def get_static_generator_config():
    generator_config = (
        (
            [
                ('conv', (1, 1, 15, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (1, 8, 31, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (8, 16, 31, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 31)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 31)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (16, 64, 15, 4, 1)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (64, 256, 7, 4, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(256, 7)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(256, 7)),
                ('lrelu', LRELU_SLOPE),
            ],
            [[('conv', (256, 128, 7, 1, 1))], [('conv', (256, 128, 7, 1, 1))]]
        ),
        (
            [[('conv', (128, 256, 7, 1, 1))], [('conv', (128, 256, 7, 1, 1))]],
            [
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(256, 7)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(256, 7)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (256, 64, 7, 4, 1)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (64, 16, 15, 4, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 31)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 31)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (16, 8, 31, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (8, 1, 31, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (1, 1, 15, 1, 1)),
                ('tanh',)
            ]
        )
    )
    return generator_config


def get_res_block_config(channel_size, kernel_size):
    common_res_blocks = \
        [
            ('sub_model',
             [
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 1)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 2)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
             ]
             ),
            ('sub_model',
             [
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 2)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 6)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
             ]
             ),
            ('sub_model',
             [
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 3)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
                 [
                     ('conv', (channel_size, channel_size, kernel_size, 12)),
                     ('conv', (channel_size, channel_size, kernel_size, 1))
                 ],
             ]
             ),
        ]
    return common_res_blocks
