import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
v=np.array([1.0, 1.0, 1.0])
a=np.array([0.0423014376934238,0.0014648828223189,0.0015049698754822])
b=np.array([0.0359501479409996,0.0197927794856654,-0.0104716367451623])


print("cos a v= ",dot(a,v)/norm(a)/norm(v))
print("cos b v= ",dot(b,v)/norm(b)/norm(v))
a=np.array([-0.75640472, 0.56845545, -0.37559479])
b=np.array([-0.9737376032934619, 0.25019030666344416, 0.1600032366300177])
print("cos a v= ",dot(a,v)/norm(a)/norm(v))
print("cos b v= ",dot(b,v)/norm(b)/norm(v))
