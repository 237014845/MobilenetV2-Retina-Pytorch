net_struct = {
    'alexnet': {'net': [[11, 4, 0], [3, 2, 0], [5, 1, 2], [3, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 2, 0]],
                'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5', 'pool5']},
    'vgg16': {'net': [[3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1], [2, 2, 0], [3, 1, 1], [3, 1, 1], [3, 1, 1],
                      [2, 2, 0]],
              'name': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'pool2', 'conv3_1', 'conv3_2',
                       'conv3_3', 'pool3', 'conv4_1', 'conv4_2', 'conv4_3', 'pool4', 'conv5_1', 'conv5_2', 'conv5_3',
                       'pool5']},
    'resnet50': {'net': [[7,2,3], [3,2,1], [1,1,0], [3,1,1], [1,1,0], [1,1,0], [3,1,1], [1,1,0], [1,1,0], [3,1,1], [1,1,0],
                         [1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],
                         [1, 1, 0], [3, 2, 1], [1, 1, 0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0],
                         [1, 1, 0], [3, 2, 1], [1, 1, 0],[1,1,0], [3,2,1], [1,1,0],[1,1,0], [3,2,1], [1,1,0], [3,2,1], [3,2,1]],
                 'name':['conv1', 'pool', 'conv1_1', 'conv1_2', 'conv1_3', 'conv2_1', 'conv2_2', 'conv2_3', 'conv3_1', 'conv3_2', 'conv3_3',
                         'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'conv6_1', 'conv6_2', 'conv6_3', 'conv7_1', 'conv7_2', 'conv7_3',
                         'conv8_1', 'conv8_2', 'conv8_3', 'conv9_1', 'conv9_2', 'conv9_3', 'conv10_1', 'conv10_2', 'conv10_3', 'conv11_1', 'conv11_2', 'conv11_3', 'conv12_1', 'conv12_2', 'conv12_3', 'conv13_1', 'conv13_2', 'conv13_3',
                         'conv14_1', 'conv14_2', 'conv14_3', 'conv15_1', 'conv15_2', 'conv15_3', 'conv16_1', 'conv16_2', 'conv16_3', 'conv17', 'conv18'
                         ]},

    'zf-5': {'net': [[7, 2, 3], [3, 2, 1], [5, 2, 2], [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1]],
             'name': ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'conv4', 'conv5']}}
imsize = 640


def outFromIn(isz, net, layernum):
    totstride = 1
    insize = isz
    for layer in range(layernum):
        fsize, stride, pad = net[layer]
        outsize = (insize - fsize + 2 * pad) / stride + 1
        insize = outsize
        totstride = totstride * stride
    return outsize, totstride


def inFromOut(net, layernum):
    RF = 1
    for layer in reversed(range(layernum)):
        fsize, stride, pad = net[layer]
        RF = ((RF - 1) * stride) + fsize
    return RF


if __name__ == '__main__':
    print("layer output sizes given image = %dx%d" % (imsize, imsize))

for net in net_struct.keys():
    print('************net structrue name is %s**************' % net)
    for i in range(len(net_struct[net]['net'])):
        p = outFromIn(imsize, net_struct[net]['net'], i + 1)
        rf = inFromOut(net_struct[net]['net'], i + 1)
        print("Layer Name = %s, Output size = %3d, Stride = % 3d, RF size = %3d" % (
        net_struct[net]['name'][i], p[0], p[1], rf))