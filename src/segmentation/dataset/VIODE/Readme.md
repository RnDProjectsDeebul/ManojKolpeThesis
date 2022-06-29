# VIODE

A novel dataset recorded from a simulated UAV that navigates in challenging dynamic environments.

- One indoor and two outdoor environment
- Each environment four sequence data are generated

Dataset can be downloaded from [A link to Zenodo](https://zenodo.org/record/4493401)
- Data is present in the .bag file format ( A bag is a file format in ROS for storing ROS message data.)

Install bagpy package
- `!pip install bagpy`

To check the topics in the bag file run the below snippet file

from bagpy import bagreader
```
b = bagreader('1_low.bag')
print(b.topic_table)
```
output:

![alt text](/image/img1.png)

To extract the image data from the bag file follow method

Requirements

- ros - [Installation instruction](http://wiki.ros.org/noetic/Installation/Ubuntu)
- Commands:
```
$sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
$sudo apt install curl # if you haven't already installed curl
$curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

$sudo apt update
$sudo apt install ros-noetic-desktop-full
$source /opt/ros/noetic/setup.bash
$echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
$source ~/.bashrc
```

### [Export image and video data from a bag file](http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data)

```
roscd image_view
rosmake image_view
sudo apt-get install mjpegtools
```
Exporting jpegs from bag file

To export jpeg images from a bag file first you will need to create a launch file which will dump the data. This example uses /camera/image_raw as the topic for the desired image data. This can be replaced as needed.
Below is the `export.launch` file
```
<launch>
  <node pkg="rosbag" type="play" name="rosbag" required="true" args="/home/latai/Downloads/data3/parking_lot/1_low.bag" />
  <node name="extract" pkg="image_view" type="extract_images" respawn="false" required="true" output="screen" cwd="ROS_HOME">
    <remap from="image" to="/cam0/image_raw"/>
  </node>
</launch>
```
```
roslaunch export.launch
```
```
cd ~
mkdir test
mv ~/.ros/frame*.jpg test/
```

Images to video

```
cd ~/test
jpeg2yuv -I p -f 15 -j frame%04d.jpg -b 1 > tmp.yuv
ffmpeg2theora --optimize --videoquality 10 --videobitrate 16778 -o output.ogv tmp.yuv
```
