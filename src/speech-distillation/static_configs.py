LRELU_SLOPE = 0.1


def get_static_all_in_one_discriminator(extra_channels=1, ensemble_size=3):
    return ('fmap',
            (
                ('ensemble',
                 [
                     get_static_single_all_in_one_discriminator_fmap(extra_channels) for i in range(ensemble_size)
                 ]
                 ),
                ['all_in_one']
            )
            )


def get_static_single_all_in_one_discriminator_fmap(extra_channels=1):
    return ('pfmap',
            (
                get_static_single_all_in_one_discriminator(extra_channels=extra_channels),
                ['all_in_one'],
                [
                    ('conv', (extra_channels, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 2, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 3, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 5, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 8, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 13, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 21, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 33, 1, 3, 1, 2)),
                    ('conv', (extra_channels * 54, 1, 3, 1, 2)),
                ]
            )
            )


def get_period_raw_block(post_scale):
    return ('period', (post_scale,)), post_scale


def get_mel_raw_block(extra_channels, post_scale):
    return (
        ('mel', (22050, post_scale, extra_channels * post_scale, post_scale)),
        post_scale
    )


def get_all_raw_blocks(extra_channels, post_scale):
    return [
        get_period_raw_block(post_scale),
        get_mel_raw_block(extra_channels, post_scale)
    ]


def get_static_single_all_in_one_discriminator(extra_channels=1):
    return ('all_in_one_discriminator',
            (
                [
                    ('conv', (1, extra_channels, 33, 1, 1, 1, 'spectral')),
                    ('lrelu', LRELU_SLOPE, ['all_in_one']),
                ],
                [
                    get_all_in_one_block_config(extra_channels, 21, 5, 1, 2, extra_channels,
                                                raw_blocks=[get_period_raw_block(2)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 2, 21, 5, 2, 3, extra_channels,
                                                raw_blocks=[get_period_raw_block(3)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 3, 13, 5, 3, 5, extra_channels,
                                                raw_blocks=[get_period_raw_block(5)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 5, 13, 5, 5, 8, extra_channels,
                                                raw_blocks=[get_period_raw_block(8)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 8, 9, 5, 8, 13, extra_channels,
                                                raw_blocks=[get_period_raw_block(13)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 13, 9, 5, 13, 21, extra_channels,
                                                raw_blocks=get_all_raw_blocks(extra_channels, 21),
                                                tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 21, 5, 5, 21, 33, extra_channels,
                                                raw_blocks=[get_period_raw_block(33)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 33, 3, 5, 33, 54, extra_channels,
                                                raw_blocks=[get_period_raw_block(54)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 54, 1, 5, 54, 90, extra_channels,
                                                raw_blocks=get_all_raw_blocks(extra_channels, 90),
                                                tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 90, 1, 5, 90, 144, extra_channels,
                                                raw_blocks=[get_period_raw_block(144)], tags=['all_in_one']),
                ],
                [
                    ('conv', (extra_channels * 144, 1, 33, 1, 1)),
                ],
            )
            )


def get_all_in_one_block_config(pre_channels, kernel_size, dilation, pre_scale, post_scale, groups=1,
                                raw_blocks=None,
                                tags=[]):
    post_channels = (pre_channels // pre_scale) * post_scale
    mid_channels = post_channels + sum(out_size for config, out_size in raw_blocks)
    raw_blocks = [config for config, out_size in raw_blocks]
    return ('all_in_one_block',
            (
                [
                    ('conv_rech', (pre_channels, post_channels, kernel_size, None, 1, groups)),
                    ('lrelu', LRELU_SLOPE),
                ],
                raw_blocks,
                [
                    ('conv', (mid_channels, post_channels, kernel_size, 1, 1, 1)),
                    ('lrelu', LRELU_SLOPE),
                    ('res',
                     ('conv', (post_channels, post_channels, kernel_size, 1, dilation, groups)),
                     ),
                    ('lrelu', LRELU_SLOPE, tags),
                ]
            )
            )


def get_static_generator_config():
    generator_config = [
        ('encoder',
         (
             [
                 ('conv', (1, 16, 25, 1, 1)),
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(16, 25)),
                 ('lrelu', LRELU_SLOPE),
                 ('conv', (16, 80, 45, 5, 1)),
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(80, 15)),
                 ('lrelu', LRELU_SLOPE),
                 ('conv', (80, 256, 15, 3, 1)),
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(256, 5)),
                 ('lrelu', LRELU_SLOPE),
             ],
             [('conv', (256, 128, 5, 1, 1)), ('conv', (256, 128, 5, 1, 1))]
         )
         ),
        ('decoder',
         (
             [('conv', (128, 256, 5, 1, 1)), ('conv', (128, 256, 5, 1, 1))],
             [
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(256, 5)),
                 ('lrelu', LRELU_SLOPE),
                 ('trans', (256, 80, 15, 3, 1)),
                 ('sub_res',
                  ('poold', (45, 1, 3))
                  ),
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(80, 15)),
                 ('lrelu', LRELU_SLOPE),
                 ('trans', (80, 80, 45, 5, 1)),
                 ('sub_res',
                  ('poold', (45, 1, 5))
                  ),
                 ('conv', (80, 16, 45, 1, 1)),
                 ('lrelu', LRELU_SLOPE),
                 ('fusion', get_res_block_config(16, 31)),
                 ('lrelu', LRELU_SLOPE),
                 ('conv', (16, 1, 31, 1, 1)),
                 ('tanh',)
             ]
         )
         )
    ]
    return generator_config


def get_res_block_config(channel_size, kernel_size):
    common_res_blocks = \
        [
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 2)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
            ],
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 2)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 6)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
            ],
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 3)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 12)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1)),
                ]),
            ],
        ]
    return common_res_blocks
