# Weather prediction from stereo images
### Project 2 in INF367A : Topological Deep Learning
**Odin Hoff Gardå, April 2023**

## Dataset

The dataset consists of 1000 stereo images (one left and one right image). Each image has 3 channels (RGB) with resolution 879x400 (w x h). The possible label values are 'cloudy' (0) and 'sunny' (1). The dataset is perfectly balanced with 500 samples of each label.

![Sunny image](figs/image_2.png)
![Cloudy image](figs/image_3.png)
*Figure: Two images (index 2 and 3) from the dataset (left and right view) with labels 'sunny' and 'cloudy'.*

## Symmetries

### Dihedral group $D_2$

The symmetry group of a rectangle is the dihedral group $D_2$, isomorphic to the Klein four-group $\mathbb{Z}_2\times\mathbb{Z}_2$. Geometrically, the group $D_2$ can be described by the following symmetries:
- $e$: rotation by 0 (identity),
- $r$: rotation by $\pi$,
- $m_h$: mirroring along the horizontal axis, and
- $m_v$: mirroring along the vertical axis.

It is easy to write down the Cayley table of $D_2$:

|$D_2$|$e$|$r$|$m_h$|$m_v$|
|:---|:---:|:---:|:---:|:---:|
|$e$|$e$|$r$|$m_h$|$m_v$|
|$r$|$r$|$e$|$m_v$|$m_h$|
|$m_h$|$m_h$|$m_v$|$e$|$r$|
|$m_v$|$m_v$|$m_h$|$r$|$e$|
