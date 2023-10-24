Environment：
registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v5.0
pip install waterz


Run:
python main.py -c=seg_3d_ac4_data80

dataset:
train: ac4 (the top 80 slices)
valid: ac4 (the bottom 20 slices)
test: ac3


valid:
179k          Iters
0.391671    Split
0.255415    Merge
0.647087    VOI
0.117131    Arand
0.50           threshold
waterz

179k
0.624914
0.177449
0.802363
0.062664
-
LMC


Test:
179k
0.659459
0.283743
0.943201
0.087484
0.5
waterz


179k
1.164816
0.193161
1.357977
0.101348
-
LMC
