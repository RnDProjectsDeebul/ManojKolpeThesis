# This is a readme for scannet dataset preperation 

Link to download the scannet dataset is given in the Re_ [ScanNet] Requesting for Scannet dataset.pdf file.

The main dataset folder is here [Link](https://github.com/ScanNet/ScanNet)

To download the entire dataset (1.3TB)

```
python download-scannet.py -o "/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/scans/"
```
Download a specific scan (e.g., scene0000_00)

```
python download-scannet.py -o "/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/scans/" --id scene0000_00
```
Download a specific file type (e.g., *.sens)

```
python download-scannet.py -o "/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/scans/" --type .sens
```

Download the ScanNet v1 task data (inc. trained models)

```
python download-scannet.py -o "/scratch/mkolpe2s/MT/Main_data_folder/Segmentation_dataset/scannet/scans/" --task_data
```

Once the dataset is downloaded , inorder to create the 2d data containing the following folder structure please follow the [Link](https://github.com/angeladai/3DMV/tree/master/prepare_data)

The downloaded data for scene0000_00 has the following file structure:
```
scene0000_00/
|--scene0000_00.sens
       ⋮
|--scene0000_00.aggregation.json
       ⋮
|--scene0000_00.txt
       ⋮
|--scene0000_00_2d-instance.zip
       ⋮
|--scene0000_00_2d-instance-filt.zip
       ⋮
|--scene0000_00_2d-label.zip
       ⋮
|--scene0000_00_2d-label-filt.zip
       ⋮
|--scene0000_00_vh_clean.aggregation.json
       ⋮
|--scene0000_00_vh_clean.ply
       ⋮
|--scene0000_00_vh_clean.segs.json
       ⋮
|--scene0000_00_vh_clean_2.labels.ply
       ⋮
|--scene0000_00_vh_clean_2.ply
       ⋮
scene0000_01/
⋮
```

Note: Unzip the .zip files and the resulting four folders are

1. instance
2. label-filt
3. label
4. instance-filt

Place the label-filt inside the instance and label folder.
Copy the `scene0000_00.sens` inside label and instance folder and rename it to `label.sens` and `instance.sens` respectively.

Now run the `prepare_2d_data.py` file. It has dependency of `SensorData.py` and `util.py`. The code is based on python 2.7. The requirements are given at the same folder level.

2D data is expected to be in the following file structure after running the `prepare_2d_data.py` file:
```
scene0000_00/
|--color/
   |--[framenum].jpg
       ⋮
|--depth/
   |--[framenum].png   (16-bit pngs)
       ⋮
|--pose/
   |--[framenum].txt   (4x4 rigid transform as txt file)
       ⋮
|--label/    (if applicable)
   |--[framenum].png   (8-bit pngs)
       ⋮
scene0000_01/
⋮
```

To get the color segmentation image from the label 8-bit pngs, find the jupyter notebook (raw_image_to_segmented_image.ipynb) given inside the `raw_to_color` folder (One folder above the current folder).  


