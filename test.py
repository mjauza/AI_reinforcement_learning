import numpy as np
import matplotlib.pyplot as plt
david1 = np.array([1,2,3,4,5])
david2 = np.array([1,2,3,4,20])
fig1 = plt.figure()
plt.plot(david1)
#plt.savefig('david1.png')

fig2 = plt.figure()
plt.plot(david2)
#plt.savefig('david2.png')

import sys
for arg in  sys.argv:
    print(arg)

from os import listdir
from os.path import isfile, join
mypath = './'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
    print(f)

