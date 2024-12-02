import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import joblib

interhand_keypoints_data = joblib.load("/hdd/zen/dev/meta/PHC_X/data/reinterhand/interhand_keypoints.pkl")
j3d = interhand_keypoints_data['m--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands']['data_array'][:, :, 1:4]/1000
idx = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(90, 0)
ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
drange = 0.5
ax.set_xlim(-drange, drange)
ax.set_ylim(-drange, drange)
ax.set_zlim(-drange, drange)
plt.show()