LRELU_SLOPE = 0.1


def get_discriminator_config(layers, expansion_size=1, ensemble_size=3):
    return ('fmap',
            (
                ('ensemble',
                 [
                     get_static_single_all_in_one_discriminator_fmap(layers, expansion_size) for i in
                     range(ensemble_size)
                 ]
                 ),
                ['all_in_one']
            )
            )


def get_discriminator_process_layer(extra_channels, channels, kernel, dilation=1, groups=1, normalization='weight'):
    return [('conv', (extra_channels * channels, 1, 3, 1, 2)), ('tanh',)]


def get_static_single_all_in_one_discriminator_fmap(layers, extra_channels=1):
    process_layers = [get_discriminator_process_layer(*layer_params) for layer_type, layer_params in layers]
    return (
        'pfmap',
        (
            get_static_single_all_in_one_discriminator(layers=layers, extra_channels=extra_channels),
            ['all_in_one'],
            process_layers
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


def get_static_single_all_in_one_discriminator(layers, extra_channels=1):
    before_layer_type, before_layer_params = layers[0]
    before_layer = get_discriminator_before_layer(extra_channels, *before_layer_params)

    in_layers = get_discriminator_in_layers(extra_channels, layers)

    after_layer_type, after_layer_params = layers[-1]
    after_layer = get_discriminator_after_layer(extra_channels, *after_layer_params)

    return (
        'all_in_one_discriminator',
        (
            before_layer,
            in_layers,
            after_layer
        )
    )


def get_discriminator_after_layer(extra_channels, channels, kernel, dilation=1, groups=1, normalization='weight'):
    after_layer = [('conv', (extra_channels * channels, 1, kernel, 1, dilation, groups, normalization)), ('tanh',)]
    return after_layer


def get_discriminator_in_layer(extra_channels, layer_type, next_channels, channels, kernel, dilation=1, groups=1):
    if layer_type[0] == 'roll':
        raw_blocks = get_roll_raw_block(next_channels)
    else:
        raw_blocks = get_all_raw_blocks(extra_channels, next_channels)
    return get_all_in_one_block_config(
        extra_channels * channels, kernel, dilation, channels, next_channels, groups,
        raw_blocks=raw_blocks, tags=['all_in_one'])


def get_discriminator_in_layers(extra_channels, layers):
    in_layers = []
    for current_index in range(1, len(layers) - 1):
        next_index = current_index + 1
        next_channels = layers[next_index][1][0]
        current_layer_type, current_layer_params = layers[current_index]
        in_layer = get_discriminator_in_layer(extra_channels, current_layer_type, next_channels, *current_layer_params)
        in_layers.append(in_layer)
    return in_layers


def get_discriminator_before_layer(extra_channels, channels, kernel, dilation=1, groups=1, normalization='spectral'):
    before_layer = [
        ('conv', (1, extra_channels * channels, kernel, 1, dilation, groups, normalization)),
        ('lrelu', LRELU_SLOPE, ['all_in_one']),
    ]
    return before_layer


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


def get_generator_configs(layers: list, expansion_size=16, embedding_size=273):
    reverse_layers = layers.copy()
    reverse_layers.reverse()
    last_encoder, last_decoder = get_last_level_model(expansion_size, embedding_size)

    current_encoder, current_decoder = last_encoder, last_decoder
    for current_index in range(0, len(reverse_layers) - 1):
        current_layer_type, current_layer_params = reverse_layers[current_index]
        current_encoder, current_decoder = get_leveln_model(
            current_encoder, current_decoder, expansion_size,
            current_layer_type, *current_layer_params
        )

    first_layer_type, first_layer_params = reverse_layers[-1]
    encoder, decoder = get_first_level_model(current_encoder, current_decoder, expansion_size,
                                             first_layer_type, *first_layer_params)
    return {'encoder': encoder, 'decoder': decoder}


def get_leveln_model(inner_encode, inner_decode, expansion, current_level_type, channels=1, kernel=63, stride=1,
                     dilation=1, groups=1):
    auto_type, upsample_type = current_level_type
    encode_block = get_block_config(auto_type, expansion, channels, kernel, stride, dilation, groups)
    decode_block = get_block_config(auto_type, expansion, channels, kernel, stride, dilation, groups)
    if upsample_type == 'sub_res':
        decode_block = [
            decode_block,
            ('sub_res',
             ('pool', (31, 1))
             ),
        ]

    encoder = [
        encode_block,
        ('roll', (stride,)),
        inner_encode
    ]
    decoder = [
        inner_decode,
        ('unroll', (stride,)),
        decode_block
    ]
    return encoder, decoder


def get_first_level_model(encoder2, decoder2, expansion_size, layer_type, channels=1, kernel=63, stride=1, dilation=1,
                          groups=1):
    en_layer = [('conv', (1, expansion_size, kernel, 1, dilation, groups, 'spectral')), ('lrelu', LRELU_SLOPE)]
    de_layer = [('conv', (expansion_size, 1, kernel, 1, dilation, groups)), ('tanh',)]

    (layer_type,) = layer_type
    if layer_type == 'res':
        en_layer = ('sum', [
            en_layer,
            ('repl', expansion_size)
        ])
        de_layer = ('sum', [
            de_layer,
            ('avg_ch',)
        ])

    encoder = [
        en_layer,
        encoder2
    ]
    decoder = [
        decoder2,
        de_layer
    ]
    return encoder, decoder


def get_last_level_model(expansion, embedding_size):
    channels = expansion * embedding_size
    encoder = ('split', {'content': channels // 2, 'style': channels // 2})
    decoder = ('merge_dict',)
    return encoder, decoder


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


def get_block_config(block_type, expansion, channel_size, kernel_size, stride, dilation, groups=1):
    expanded_size = channel_size * expansion
    block = [
        ('conv', (expanded_size, expanded_size, kernel_size, 1, dilation, groups)),
        ('lrelu', LRELU_SLOPE),
        ('conv_shuffle', (expanded_size, expanded_size, kernel_size, 1, dilation, groups)),
        ('lrelu', LRELU_SLOPE)
    ]
    if block_type == 'res':
        block = ('res', block)
    return block


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
    return [[('conv', (layers[i], layers[i + 1], 3, 1, 1, groups[i % len(groups)])), ('lrelu', LRELU_SLOPE)] for i in
            range(len(layers) - 1)]


def generate_sniffer_config_by_example(key, label_group, example_item, hiddens=[1092, 546, 364], groups=[1],
                                       one_hot=False):
    input_channels = sum(len(value) for value in label_group.values())
    other_label_groups = {ex_key: {key2: len(value2) for key2, value2 in value.items()} for ex_key, value in
                          example_item.items() if ex_key != key}
    other_groups_channels = {
        ex_key: sum(value for value in other_label_group.values())
        for ex_key, other_label_group in other_label_groups.items()
    }
    output_channels = sum(other_groups_channels.values())
    layers = [
        ('merge_dict',),
        get_classifier_backbone(input_channels, output_channels, hiddens, groups=groups),
        ('split', other_groups_channels),
        ('recursive', {group: ('split', sizes) for group, sizes in other_label_groups.items()}),
        ('recursive', {group: {key: ('softmax',) for key in sizes} for group, sizes in other_label_groups.items()}),
    ]
    if one_hot:
        one_hot_layer = ('recursive', {label: ('one_hot', (value, 1)) for label, value in label_group.items()})
        layers = [one_hot_layer, *layers]
    return layers


def generate_sniffers_configs_by_example(example_item, hiddens=[1092, 546, 364], groups=[1], ensemble_size=3,
                                         one_hot=False):
    sniffers = {
        key: ('ensemble', [
            generate_sniffer_config_by_example(key, label_group, example_item, hiddens=hiddens, groups=[1],
                                               one_hot=one_hot)
            for i in range(ensemble_size)
        ])
        for key, label_group in example_item.items()
    }
    return sniffers
