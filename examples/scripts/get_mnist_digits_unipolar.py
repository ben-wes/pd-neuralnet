import idx2numpy, random, sys
from time import sleep

def main():
    dir = sys.argv[1] + '/data/'
    dataset = sys.argv[2]

    prefix = 'train-' if dataset == 'train' else 't10k-'

    images_file = dir + prefix + 'images.idx3-ubyte'
    labels_file = dir + prefix + 'labels.idx1-ubyte'

    images = idx2numpy.convert_from_file(images_file)
    labels = idx2numpy.convert_from_file(labels_file)

    order = [i for i in range(len(labels))]
    if dataset == 'train':
        random.shuffle(order)

    for i, idx in enumerate(order):
        img = images[idx]
        label = labels[idx]
        for row in img:
            for color in row:
                # normalize color to range 0..1
                color = color / 255
                print(f'X {color}')
        print(f'y {label}')
        print(f'row {i}')
        sleep(0.001)

if __name__ == "__main__":
    main()
