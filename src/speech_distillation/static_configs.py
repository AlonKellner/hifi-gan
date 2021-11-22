LRELU_SLOPE = 0.1


def get_static_all_in_one_discriminator(expansion_size=1, ensemble_size=3):
    return ('pfmap',
            (
                ('ensemble',
                 [
                     get_static_single_all_in_one_discriminator_fmap(expansion_size) for i in range(ensemble_size)
                 ]
                 ),
                ['all_in_one'],
                [('tanh',)]
            )
            )


def get_static_single_all_in_one_discriminator_fmap(extra_channels=1):
    return ('pfmap',
            (
                get_static_single_all_in_one_discriminator(extra_channels=extra_channels),
                ['all_in_one'],
                [
                    [('conv', (extra_channels, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 2, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 3, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 5, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 8, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 13, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 21, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 33, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 54, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 90, 1, 3, 1, 2)), ('tanh',)],
                    [('conv', (extra_channels * 144, 1, 3, 1, 2)), ('tanh',)],
                ]
            )
            )


def get_roll_raw_block(post_scale):
    return ('roll', (post_scale,)), post_scale


def get_mel_raw_block(extra_channels, post_scale):
    return (
        ('mel', (22050, post_scale, extra_channels * post_scale, post_scale)),
        post_scale
    )


def get_all_raw_blocks(extra_channels, post_scale):
    return [
        get_roll_raw_block(post_scale),
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
                                                raw_blocks=[get_roll_raw_block(2)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 2, 21, 5, 2, 3, extra_channels,
                                                raw_blocks=[get_roll_raw_block(3)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 3, 13, 5, 3, 5, extra_channels,
                                                raw_blocks=[get_roll_raw_block(5)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 5, 13, 5, 5, 8, extra_channels,
                                                raw_blocks=[get_roll_raw_block(8)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 8, 9, 5, 8, 13, extra_channels,
                                                raw_blocks=[get_roll_raw_block(13)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 13, 9, 5, 13, 21, extra_channels,
                                                raw_blocks=get_all_raw_blocks(extra_channels, 21),
                                                tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 21, 5, 5, 21, 33, extra_channels,
                                                raw_blocks=[get_roll_raw_block(33)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 33, 3, 5, 33, 54, extra_channels,
                                                raw_blocks=[get_roll_raw_block(54)], tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 54, 1, 5, 54, 90, extra_channels,
                                                raw_blocks=get_all_raw_blocks(extra_channels, 90),
                                                tags=['all_in_one']),
                    get_all_in_one_block_config(extra_channels * 90, 1, 5, 90, 144, extra_channels,
                                                raw_blocks=[get_roll_raw_block(144)], tags=['all_in_one']),
                ],
                [('conv', (extra_channels * 144, 1, 33, 1, 1)), ('tanh',)],
            )
            )


def get_all_in_one_block_config(pre_channels, kernel_size, dilation, pre_scale, post_scale, groups=1,
                                raw_blocks=None,
                                tags=[]):
    post_channels = (pre_channels // pre_scale) * post_scale
    mid_channels = post_channels + sum(out_size for config, out_size in raw_blocks)
    raw_blocks = [config for config, out_size in raw_blocks]
    mid_groups = groups if mid_channels % groups == 0 else 1
    return ('all_in_one_block',
            (
                [
                    ('conv_rech', (pre_channels, post_channels, kernel_size, None, 1, groups)),
                    ('lrelu', LRELU_SLOPE),
                ],
                raw_blocks,
                [
                    ('conv_shuffle', (mid_channels, post_channels, kernel_size, 1, 1, mid_groups)),
                    ('lrelu', LRELU_SLOPE),
                    ('res',
                     ('conv_shuffle', (post_channels, post_channels, kernel_size, 1, dilation, groups)),
                     tags),
                    ('lrelu', LRELU_SLOPE),
                ]
            )
            )


def get_static_generator_config(initial_skip_ratio=1, expansion_size=16):
    embedding_tags = ['embeddings']
    level5 = get_level5_model(initial_skip_ratio, 273 * expansion_size, embedding_tags=embedding_tags)
    level4 = get_leveln_model(initial_skip_ratio, 'skip4', 'noise4', (21 * expansion_size, 13, 13, expansion_size),
                              (273 * expansion_size, 3, 39), 31, level5)
    level3 = get_leveln_model(initial_skip_ratio, 'skip3', 'noise3', (3 * expansion_size, 21, 7, 1),
                              (21 * expansion_size, 13, 3), 31, level4)
    level2 = get_leveln_model(initial_skip_ratio, 'skip2', 'noise2', (expansion_size, 33, 3, 1),
                              (3 * expansion_size, 21, 1), 31, level3)
    generator_config = get_level1_model(initial_skip_ratio, level2, expansion_size)
    generator_config = ('fmap', (generator_config, embedding_tags))
    return generator_config


def get_leveln_model(initial_skip_ratio, skip_tag, anti_tag, out_params, mid_params, pool_reception, next_model):
    out_channels, out_kernel, out_stride, out_groups = out_params
    mid_channels, mid_kernel, mid_groups = mid_params
    return \
        get_decaying_block(
            initial_skip_ratio, skip_tag, anti_tag, out_channels,
            [
                # ('conv', (out_channels, mid_channels, out_kernel, out_stride, 1, out_groups)),
                ('roll', (out_stride,)),
                get_res_block_config(mid_channels, mid_kernel, mid_groups),
                ('lrelu', LRELU_SLOPE),
                next_model,
                ('lrelu', LRELU_SLOPE),
                get_res_block_config(mid_channels, mid_kernel, mid_groups),
                ('lrelu', LRELU_SLOPE),
                ('sub_res',
                 ('pool', (pool_reception, 1))
                 ),
                ('unroll', out_stride),
                # ('trans', (mid_channels, out_channels, out_kernel, out_stride, 1, out_groups)),
            ]
        )


def get_level1_model(initial_skip_ratio, level2, expansion_size=16):
    return \
        [
            ('sum', [
                ('conv', (1, expansion_size, 63, 1, 1, 1, 'spectral')),
                ('repl', expansion_size)
            ]),
            get_decaying_block(
                initial_skip_ratio, 'skip1', 'noise1', expansion_size,
                [
                    ('lrelu', LRELU_SLOPE),
                    get_res_block_config(expansion_size, 33),
                    ('lrelu', LRELU_SLOPE),
                    level2,
                    ('lrelu', LRELU_SLOPE),
                    get_res_block_config(expansion_size, 33),
                    ('lrelu', LRELU_SLOPE),
                ]
            ),
            ('sum', [
                ('conv', (expansion_size, 1, 63, 1, 1)),
                ('avg_ch',)
            ]),
            ('tanh',)
        ]


def get_level5_model(initial_skip_ratio, channels, embedding_tags=[]):
    return \
        get_decaying_block(
            initial_skip_ratio, 'skip5', 'noise5', channels,
            [
                ('split', {'content': channels // 2, 'style': channels // 2}, embedding_tags),
                ('merge_dict',)
            ]
        )


def get_decaying_block(initial_skip_ratio, skip_tag, anti_tag, noise_channels, inner_block):
    if initial_skip_ratio <= 0:
        return inner_block
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
    # return 'res', ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups))
    return ('res', [
        ('conv', (channel_size, channel_size, kernel_size, 1, 1, groups)),
        ('lrelu', LRELU_SLOPE),
        ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
    ])
    # return [
    #         ('res', [
    #             ('conv', (channel_size, channel_size, kernel_size, 1, 1, groups)),
    #             ('lrelu', LRELU_SLOPE),
    #             ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
    #         ]),
    #         ('res', [
    #             ('conv', (channel_size, channel_size, kernel_size, 1, 1, groups)),
    #             ('lrelu', LRELU_SLOPE),
    #             ('conv_shuffle', (channel_size, channel_size, kernel_size, 1, 1, groups)),
    #         ]),
    #     ]


def get_fusion_res_block_config(channel_size, kernel_size, groups=1):
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
    return 'fusion', common_res_blocks


def get_classifier_backbone(input_channels, output_channels, hiddens, groups=[1]):
    layers = [input_channels, *hiddens, output_channels]
    return [('conv', (layers[i], layers[i + 1], 3, 1, 1, groups[i % len(groups)])) for i in range(len(layers) - 1)]


def generate_sniffer_config_by_example(key, label_group, example_item, hiddens=[1092, 546, 364], one_hot=False):
    input_channels = sum(value for value in label_group.values())
    other_label_groups = {ex_key: value for ex_key, value in example_item.items() if ex_key != key}
    other_groups_channels = {
        ex_key: sum(value for value in other_label_group.values())
        for ex_key, other_label_group in other_label_groups.items()
    }
    output_channels = sum(other_groups_channels.values())
    layers = [
        ('merge_dict',),
        get_classifier_backbone(input_channels, output_channels, hiddens),
        ('split', other_groups_channels),
        ('recursive', {group: ('split', sizes) for group, sizes in other_label_groups.items()}),
        ('recursive', {group: {key: ('softmax',) for key in sizes} for group, sizes in other_label_groups.items()}),
    ]
    if one_hot:
        one_hot_layer = ('recursive', {label: ('one_hot', (value, 1)) for label, value in label_group.items()})
        layers = [one_hot_layer, *layers]
    return layers


def generate_sniffers_configs_by_example(example_item, hiddens=[1092, 546, 364], ensemble_size=3, one_hot=False):
    sniffers = {
        key: ('ensemble', [
            generate_sniffer_config_by_example(key, label_group, example_item, hiddens=hiddens, one_hot=one_hot)
            for i in range(ensemble_size)
        ])
        for key, label_group in example_item.items()
    }
    return sniffers
