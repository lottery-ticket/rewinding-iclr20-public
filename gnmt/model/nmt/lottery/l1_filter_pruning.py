vgg_16_A = {
    'conv_1': 0.5,
    'conv_2': 0.5,
    'conv_3': 0,
    'conv_4': 0,
    'conv_5': 0,
    'conv_6': 0,
    'conv_7': 0,
    'conv_8': 0.5,
    'conv_9': 0.75,
    'conv_10': 0.75,
    'conv_11': 0.75,
    'conv_12': 0.75,
    'conv_13': 0.75,
}


resnet_56_A = {
    'conv_{}'.format(i): 0.1 for i in range(2, 55, 2) if i not in (16, 20, 38, 54)
}

resnet_56_B = {
    'conv_{}'.format(i): (
        0.6 if i <= 19 else
        0.3 if i <= 37 else
        0.1
    ) for i in range(2, 56, 2) if i not in (16, 18, 20, 34, 38, 54)
}

resnet_110_A = {
    'conv_{}'.format(i): 0.5 for i in range(2, 37, 2) if i not in (36,)
}

resnet_110_B = {
    'conv_{}'.format(i): (
        0.5 if i <= 37 else
        0.4 if i <= 73 else
        0.3
    ) for i in range(2, 110, 2) if i not in (36, 38, 74)
}

resnet_34_A = {
    'conv_{}'.format(i): (
        0.3 if i <= 6 else
        0.3 if i <= 14 else
        0.3
    ) for i in range(2, 27, 2) if i not in (2, 8, 14, 16, 26, 28, 30, 32)
}

resnet_34_B = {
    'conv_{}'.format(i): (
        0.5 if i <= 6 else
        0.6 if i <= 14 else
        0.4
    ) for i in range(2, 27, 2) if i not in (2, 8, 14, 16, 26, 28, 30, 32)
}
