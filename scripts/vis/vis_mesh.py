import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as sRot
mesh_data_grab = o3d.io.read_triangle_mesh("phc/data/assets/mesh/grab/flashlight.stl")
mesh_data_grab = mesh_data_grab.rotate(sRot.from_euler("xyz", [np.pi/3, 0, 0]).as_matrix(), center = (0, 0, 0))
j3d = np.asarray(mesh_data_grab.vertices)[None]


idx = 0
fig = plt.figure(dpi = 500)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(0, 0)
ax.scatter(j3d[idx, :,0], j3d[idx, :,1], j3d[idx, :,2], s = 0.001)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
drange = 0.1
ax.set_xlim(-drange, drange)
ax.set_ylim(-drange, drange)
ax.set_zlim(-drange, drange)
ax.set_axis_off()
plt.show()