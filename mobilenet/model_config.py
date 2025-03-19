

MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 3, 0, True, 1, 4, False],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [96,  128, 3, 3, True, 2, 6, False],
            [128, 128, 5, 5, True, 1, 4, False],
            [128, 128, 0, 5, True, 1, 4, False],
            [128, 128, 0, 5, True, 1, 3, False],
            [128, 128, 0, 3, True, 1, 4, False],
            [128, 128, 0, 3, True, 1, 4, False],
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [128, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4, False],
            [80, 80, 3, 3, True, 1, 2, False]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80,  160, 3, 5, True, 2, 6, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 5, True, 1, 4, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 0, True, 1, 4, False],
            [160, 160, 0, 0, True, 1, 2, False],
            [160, 160, 3, 0, True, 1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6, False],
            [256, 256, 5, 5, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 3, 0, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 2, False],
            [256, 256, 5, 5, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 5, 0, True, 1, 2, False]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [256, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96,  192, 3, 5, True, 2, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 5, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 3, 0, True, 1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 13,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 3, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 3, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [512, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


def mhsa(num_heads, key_dim, value_dim, px):
    if px == 24:
        kv_strides = 2
    elif px == 12:
        kv_strides = 1
    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True
    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


MNV4HybridConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4, True],
            [80, 80, 3, 3, True, 1, 2, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80,  160, 3, 5, True, 2, 6, True],
            [160, 160, 0, 0, True, 1, 2, True],
            [160, 160, 3, 3, True, 1, 4, True],
            [160, 160, 3, 5, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 12,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6, True],
            [256, 256, 5, 5, True, 1, 4, True],
            [256, 256, 3, 5, True, 1, 4, True],
            [256, 256, 3, 5, True, 1, 4, True],
            [256, 256, 0, 0, True, 1, 2, True],
            [256, 256, 3, 5, True, 1, 2, True],
            [256, 256, 0, 0, True, 1, 2, True],
            [256, 256, 0, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 3, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 5, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, True]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [256, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4HybridConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, False, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4, True],
            [96, 96, 3, 3, True, 1, 4, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96,  192, 3, 5, True, 2, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 5, True, 1, 4, True],
            [192, 192, 5, 3, True, 1, 4, True],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 3, 0, True, 1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 14,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 3, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 3, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [512, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS
}