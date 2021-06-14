imshape = (256, 256, 3)

mode = 'multi'

model_name = 'fcn_8_'+mode

logbase = 'logs'

hues = {'star': 30,
        'square': 0,
        'circle': 90,
        'triangle': 60}

labels = sorted(hues.keys())

if mode == 'binary':
    n_classes = 1

elif mode == 'multi':
    n_classes = len(labels) + 1
