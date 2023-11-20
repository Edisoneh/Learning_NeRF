# Learning_NeRF
At the beginning of the study, I will reproduce the problems encountered by NeRF and sort them out under this project. 
The basic framework for the reproduction is based on the framework code of https://github.com/pengsida/learning_nerf.
The dataset can be obtained at: [https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1].
Emphasis: For the reproduction of each framework process, it is highly recommended to use 
### debug
```
python run.py --type xxxx (can be dataset, etc.) --cfg_file configs/xxx/xxx.yaml
```
For example, for the dataset debug in the NeRF reproduction process, you can use
### debug
```
python run.py --type dataset --cfg_file configs/nerf/nerf.yaml
```
According to [https://github.com/pengsida/learning_nerf/blob/master/README.md].
