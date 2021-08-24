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


def get_static_generator_config(initial_skip_ratio=1):
    level5 = get_level5_model(initial_skip_ratio)
    level4 = get_leveln_model(initial_skip_ratio, 'skip4', 'noise4', (336, 13, 13, 16), (4368, 3, 56), 31, level5)
    level3 = get_leveln_model(initial_skip_ratio, 'skip3', 'noise3', (48, 21, 7, 1), (336, 13, 3), 31, level4)
    level2 = get_leveln_model(initial_skip_ratio, 'skip2', 'noise2', (16, 33, 3, 1), (48, 21, 1), 31, level3)
    generator_config = get_level1_model(initial_skip_ratio, level2)
    return generator_config


def get_leveln_model(initial_skip_ratio, skip_tag, anti_tag, out_params, mid_params, pool_reception, next_model):
    out_channels, out_kernel, out_stride, out_groups = out_params
    mid_channels, mid_kernel, mid_groups = mid_params
    return \
        get_decaying_block(
            initial_skip_ratio, skip_tag, anti_tag, out_channels,
            [
                ('conv', (out_channels, mid_channels, out_kernel, out_stride, 1, out_groups)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(mid_channels, mid_kernel, mid_groups)),
                ('lrelu', LRELU_SLOPE),
                next_model,
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(mid_channels, mid_kernel, mid_groups)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (mid_channels, out_channels, out_kernel, out_stride, 1, out_groups)),
                ('sub_res',
                 ('poold', (pool_reception, 1, out_stride))
                 ),
            ]
        )


def get_level1_model(initial_skip_ratio, level2=None):
    if level2 is None:
        level2 = get_level2_model(initial_skip_ratio)
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip1', 'noise1', 1,
            [
                ('conv', (1, 16, 63, 1, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 33)),
                ('lrelu', LRELU_SLOPE),
                level2,
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(16, 33)),
                ('lrelu', LRELU_SLOPE),
                ('conv', (16, 1, 63, 1, 1)),
                ('tanh',)
            ]
        )


def get_level2_model(initial_skip_ratio):
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip2', 'noise2', 16,
            [
                ('conv', (16, 48, 33, 3, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(48, 21)),
                ('lrelu', LRELU_SLOPE),
                get_level3_model(initial_skip_ratio),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(48, 21)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (48, 16, 33, 3, 1)),
                ('sub_res',
                 ('poold', (31, 1, 3))
                 ),
            ]
        )


def get_level3_model(initial_skip_ratio):
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip3', 'noise3', 48,
            [
                ('conv', (48, 336, 21, 7, 1)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(336, 13, 3)),
                ('lrelu', LRELU_SLOPE),
                get_level4_model(initial_skip_ratio),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(336, 13, 3)),
                ('lrelu', LRELU_SLOPE),
                ('trans', (336, 48, 21, 7, 1)),
                ('sub_res',
                 ('poold', (31, 1, 7))
                 ),
            ]
        )


def get_level4_model(initial_skip_ratio):
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip4', 'noise4', 336,
            [
                ('conv_shuffle', (336, 4368, 13, 13, 1, 16)),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(4368, 3, 56)),
                ('lrelu', LRELU_SLOPE),
                get_level5_model(initial_skip_ratio),
                ('lrelu', LRELU_SLOPE),
                ('fusion', get_res_block_config(4368, 3, 56)),
                ('lrelu', LRELU_SLOPE),
                ('trans_shuffle', (4368, 336, 13, 13, 1, 16)),
                ('sub_res',
                 ('poold', (31, 1, 13))
                 ),
            ]
        )


def get_level5_model(initial_skip_ratio):
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip5', 'noise5', 4368,
            [
                ('split', [
                    ('conv_shuffle', (4368, 2184, 3, 1, 1, 21)),
                    ('conv_shuffle', (4368, 2184, 3, 1, 1, 21))
                ]),
                ('merge', [
                    ('conv_shuffle', (2184, 4368, 3, 1, 1, 21)),
                    ('conv_shuffle', (2184, 4368, 3, 1, 1, 21))
                ])
            ]
        )


def get_decaying_block(initial_skip_ratio, skip_tag, anti_tag, noise_channels, inner_block):
    return \
        (
            'sum',
            [
                [
                    ('sum',
                     [
                         ('valve', initial_skip_ratio, [skip_tag]),
                         [
                             ('noise', noise_channels),
                             ('valve', 0, [anti_tag]),
                         ]
                     ]),
                    ('valve', initial_skip_ratio, [skip_tag]),
                ],
                inner_block
            ]
        )


def get_res_block_config(channel_size, kernel_size, groups=1):
    common_res_blocks = \
        [
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 2, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
            ],
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 2, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 6, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
            ],
            [
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 3, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
                ('res', [
                    ('lrelu', LRELU_SLOPE),
                    ('conv', (channel_size, channel_size, kernel_size, 1, 12, groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
                ]),
            ],
        ]
    return common_res_blocks
