#  https://stackoverflow.com/questions/17705621/algorithm-for-a-geodesic-sphere

import numpy as np

class Vector(object):
   def __init__(self, x, y, z):
      self.x = x
      self.y = y
      self.z = z

   def normalize(self):
      n = self.norm()
      return Vector(self.x/n, self.y/n, self.z/n)

   def __add__(self, v):
      return Vector(v.x + self.x, v.y + self.y, v.z + self.z)

   def norm(self):
      return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

   def dot(self, v):
      return self.x*v.x + self.y*v.y + self.z*v.z


def subdivide(v1, v2, v3, sphere_points, depth):
   if depth == 0:
      sphere_points.append(v1);
      sphere_points.append(v2);
      sphere_points.append(v3);
      return
   else:
      v12 = (v1 + v2).normalize()
      v23 = (v2 + v3).normalize()
      v31 = (v3 + v1).normalize()
      subdivide(v1, v12, v31, sphere_points, depth - 1)
      subdivide(v2, v23, v12, sphere_points, depth - 1)
      subdivide(v3, v31, v23, sphere_points, depth - 1)
      subdivide(v12, v23, v31, sphere_points, depth - 1)


def initialize_sphere(depth):
   '''
   Create a geodesic sphere of triangles, starting with a icosahedron and subdividing the 
   edges depth times.

   returns a list of Vector of the vertices of the triangles
   '''
   sphere_points = []
   X = 0.525731112119133606
   Z = 0.850650808352039932
   vdata = [
        Vector(-X, 0.0, Z), Vector( X, 0.0, Z ), Vector( -X, 0.0, -Z ), Vector( X, 0.0, -Z ),
        Vector( 0.0, Z, X ), Vector( 0.0, Z, -X ), Vector( 0.0, -Z, X ), Vector( 0.0, -Z, -X ),
        Vector( Z, X, 0.0 ), Vector( -Z, X, 0.0 ), Vector( Z, -X, 0.0 ), Vector( -Z, -X, 0.0 )]

   tindices = [
        (0, 4, 1), ( 0, 9, 4 ), ( 9, 5, 4 ), ( 4, 5, 8 ), ( 4, 8, 1 ),
        ( 8, 10, 1 ), ( 8, 3, 10 ), ( 5, 3, 8 ), ( 5, 2, 3 ), ( 2, 7, 3 ),
        ( 7, 10, 3 ), ( 7, 6, 10 ), ( 7, 11, 6 ), ( 11, 0, 6 ), ( 0, 1, 6 ),
        ( 6, 1, 10 ), ( 9, 0, 11 ), ( 9, 11, 2 ), ( 9, 2, 5 ), ( 7, 2, 11 )]

   for i in range(20):
      subdivide(vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], sphere_points, depth)

   return sphere_points


def vecs_to_lists(vecs):
   x = [v.x for v in vecs]
   y = [v.y for v in vecs]
   z = [v.z for v in vecs]
   return x, y, z



def sphere(N):
   x = np.array([])
   y = np.array([])
   z = np.array([])

   for t in np.linspace(0, 1, N):
      theta = np.pi * t # from 0 to pi
      N_ring = int(N * np.abs(np.sin(theta)))
      # d_angle = 2*np.pi / N_ring/ 2.0
      # phis_ring = np.linspace(d_angle, 2*np.pi + d_angle, N_ring)
      phis_ring = np.linspace(0, 2 * np.pi, N_ring, endpoint=False)
      tmp1_x = np.zeros(N_ring)
      tmp1_y = np.zeros(N_ring)
      tmp1_z = np.zeros(N_ring)
      for i, phi_ring in enumerate(phis_ring):
            tmp1_x[i] = np.sin(theta) * np.cos(phi_ring)
            tmp1_y[i] = np.sin(theta) * np.sin(phi_ring)
            tmp1_z[i] = np.cos(theta)
      x = np.append(x, tmp1_x)
      y = np.append(y, tmp1_y)
      z = np.append(z, tmp1_z)
   return x,y,z


# test stuff

# import matplotlib.pyplot as plt

# def test_sphere(N):
#    import matplotlib.pyplot as plt
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    x,y,z = sphere(N)
#    ax.scatter(xs=x, ys=y, zs=z, zdir='z', s=2, c=None, depthshade=True)
#    return fig


# def test_geosphere(depth):
#    s = initialize_sphere(depth)
#    print(np.arccos(s[6].dot(s[7])) / np.pi * 180)
#    x, y, z = vecs_to_lists(s)
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot(x[3:18],y[3:18],z[3:18])
#    return fig
