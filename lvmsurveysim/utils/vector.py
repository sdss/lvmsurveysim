import numpy

x1 = numpy.array([0,1,2])
x2 = numpy.array([[65,1,1],[32,56,82]])

d = numpy.sum((x2**2 - x1**2), axis=1)**0.5

u_vec = (x2-x1)/d[:,None]

print(u_vec)

d_new = numpy.full(len(u_vec), 100.0)
new = u_vec*d_new[:,None] + x1

print(new)

import lvmsurveysim.utils.shadow_height_lib as shadow_height_lib

