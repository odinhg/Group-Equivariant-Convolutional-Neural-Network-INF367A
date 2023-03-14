# Weather prediction from stereo images
### Project 2 in INF367A : Topological Deep Learning
**Odin Hoff Gardå, April 2023**

## Dataset

The dataset consists of 1000 stereo images (one left and one right image). Each image has 3 channels (RGB) with resolution 879x400 (w x h). The possible label values are 'cloudy' (0) and 'sunny' (1). The dataset is perfectly balanced with 500 samples of each label.

![Sunny image](figs/image_2.png)
![Cloudy image](figs/image_3.png)
*Figure: Two images (index 2 and 3) from the dataset (left and right view) with labels 'sunny' and 'cloudy'.*
