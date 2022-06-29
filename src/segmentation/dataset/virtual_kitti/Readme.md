# [Virtual Kitti 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)

Virtual KITTI 2 is a more photo-realistic and better-featured version of the original virtual KITTI dataset. It exploits recent improvements of the Unity game engine and provides new data such as stereo images or scene flow.

- All RGB images are encoded as RGB JPG files with 8-bit per channel.
- All class segmentation images are encoded as RGB PNG files with 8-bit per channel.
- The system of 3D camera coordinates x is going to the right, y is going down, and z is going forward (the origin is the optical center of the camera).
- 15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning, overcast, rain, sunset.
- 15 classes: 
Terrain, Sky, Tree, Vegetation, Building, Road, GuardRail, TrafficSign, TrafficLight, Pole, Misc, Truck, Car, Van, Undefined.

Download the following files

```
vkitti_2.0.3_rgb.tar
vkitti_2.0.3_classSegmentation.tar
vkitti_2.0.3_textgt.tar.gz
```
Each data folder contains two files 'camera_01' and 'camera_02'

The classSegmentation.tar file contains the segmentation label in rgb format.

The color values for different labels are as shown below

- Category r g b
- Terrain 210 0 200
- Sky 90 200 255
- Tree 0 199 0
- Vegetation 90 240 0
- Building 140 140 140
- Road 100 60 100
- GuardRail 250 100 255
- TrafficSign 255 255 0
- TrafficLight 200 200 0
- Pole 255 130 0
- Misc 80 80 80
- Truck 160 60 60
- Car 255 127 80
- Van 0 139 139
- Undefined 0 0 0

However for the unet model needs the mask i.e each image having values ranging from 0 to 15. So we need to map the rgb values to a class value as per the above values.

To map the values follow the below snippet code. Jupyter Notebook `rgb_to_mask.ipynb`

```
import os

def test():
    mapping = { (0, 0, 0): 0,        # 0 = Undefined
                (210, 0, 200): 1,    # 1 = Terrain
                (90, 200, 255): 2,   # 2 = Sky
                (0, 199, 0): 3,      # 3 = Tree
                (90, 240, 0): 4,     # 4 = Vegetation
                (140, 140, 140): 5,  # 5 = Building
                (100, 60, 100): 6,   # 6 = Road
                (250, 100, 255): 7,  # 7 = GuardRail
                (255, 255, 0): 8,    # 8 = TrafficSign
                (200, 200, 0): 9,    # 9 = TrafficLight
                (255, 130, 0): 10,   # 10 = Pole
                (80, 80, 80): 11,    # 11 = Misc
                (160, 60, 60): 12,   # 12 = Truck
                (255, 127, 80): 13,  # 13 = Car
                (0, 139, 139): 14}   # 14 = Van

    for entry in os.scandir('/home/latai/Documents/Master_thesis_v2/data/test5/class_segmentation/'):

        if os.path.isdir(entry.path) == True:
            for entry2 in os.scandir(entry.path):
                if os.path.isdir(entry2.path) == True:
                    for entry3 in os.scandir(entry2.path):
                        for entry4 in os.scandir(entry3.path):
                            print(entry4.path)
                            for entry5 in os.scandir(entry4.path):
                                
                                if entry5.path.split('/')[-1] in ['Camera_0', 'Camera_1']:
                                    directory = entry5.path.split('/')[-1]+'_n'
                                    parent_dir = entry4.path

                                    path = os.path.join(parent_dir, directory)

                                    if os.path.exists(path) is False:
                                        new_path = os.mkdir(path)
                                    for entry6 in os.scandir(entry5.path):
                                        im = Image.open(entry6.path)
                                        im = list(im.getdata())
                                        mask = []
                                        for i in im:
                                            mask.append(mapping[i])
                                        A = np.array(mask)
                                        B = np.reshape(A, (375, 1242))
                                        image = Image.fromarray(np.uint8(B), 'L')
                                        image.save(path+'/'+entry6.path.split('/')[-1])
```

## To prepare the pose values please follow the below snippet code
Same is implemented in jupyter notebook and can be found at `Pose_estimation_visualization.ipynb`

```
data = pd.read_csv('/home/latai/Documents/Master_thesis_v2/data/test5/data_prep/camera_parameters/extrinsic.txt', sep=" ")
data = data.drop_duplicates(subset='frame', keep="last")

data['frame'] = data['frame'].astype(str)
data['frame'] = data['frame'].str.zfill(5)
frames = data['frame'].tolist()
data = data.astype(float)
trunc = lambda x: math.trunc(1000000 * x) / 1000000

data = data.applymap(trunc)

for i, j in zip(range(len(data)), frames ):
    l1 = str(data.iloc[i]['r1,1'])+' '+str(data.iloc[i]['r1,2'])+' '+str(data.iloc[i]['r1,3'])+' '+str(data.iloc[i]['t1'])
    l2 = str(data.iloc[i]['r2,1'])+' '+str(data.iloc[i]['r2,2'])+' '+str(data.iloc[i]['r2,3'])+' '+str(data.iloc[i]['t2'])
    l3 = str(data.iloc[i]['r3,1'])+' '+str(data.iloc[i]['r3,2'])+' '+str(data.iloc[i]['r3,3'])+' '+str(data.iloc[i]['t3'])
    l4 = str(float(data.iloc[i]['0']))+' '+str(float(data.iloc[i]['0.1']))+' '+str(float(data.iloc[i]['0.2']))+' '+str(float(data.iloc[i]['1']))

    lines = [l1, l2, l3, l4]
    with open('/home/latai/Documents/Master_thesis_v2/data/test5/data_prep/camera_parameters/extrinsic/'+j+'.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

```

Unet expect input image of the following dimension: 1239x375, original image dimension 1242x375