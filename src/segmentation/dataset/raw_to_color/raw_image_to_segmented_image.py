import numpy as np
import imageio
import os
import csv

import os, sys, argparse


def main(input_file,label_map_file, output_file ):
    image = np.array(imageio.imread(input_file))
    label_map = read_label_mapping(label_map_file, label_from='id', label_to='nyu40id')
    mapped_image = map_label_image(image, label_map)
    # imageio.imwrite(output_file, mapped_image)
    visualize_label_image(os.path.splitext(output_file)[0] + '_vis.jpg', mapped_image)

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.items():
        mapped[image==k] = v
    return mapped.astype(np.uint8)

# if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return 
    
def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
#         print(reader)
        for row in reader:
#             print(row)
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
#     print(mapping.keys()[0])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

# color by label
def visualize_label_image(filename, image):
    height = image.shape[0]
    width = image.shape[1]
    vis_image = np.zeros([height, width, 3], dtype=np.uint8)
    color_palette = create_color_palette()
    for idx, color in enumerate(color_palette):
        vis_image[image==idx] = color
    imageio.imwrite(filename, vis_image)

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),
       (152, 223, 138),
       (31, 119, 180), 
       (255, 187, 120),
       (188, 189, 34), 
       (140, 86, 75),  
       (255, 152, 150),
       (214, 39, 40),  
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144)
    ]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='path to input label image folder')
    parser.add_argument('--label_map_file', required=True, help='path to scannetv2-labels.combined.tsv')
    parser.add_argument('--output_file', required=True, help='output image file folder')
    opt = parser.parse_args()

    for folder in os.listdir(opt.input_file):
        for file in os.listdir(opt.input_file+folder+'/'):
            if file.endswith(".png"):
                print(opt.input_file+folder+'/'+file)
                main(opt.input_file+folder+'/'+file, opt.label_map_file, opt.output_file+file)