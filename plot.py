from matplotlib import pyplot
import matplotlib as mpl
import numpy as np

# change zvals  
zvals = np.array([ 0.59049, 0.6561, 0.729, 0.6561, 0.6561, 0.0, 0.81, 0., 0.729, 0.81, 0.9, 0.0, 0.0, 0.9, 1.0, 0.0]) 
# resize zvals to (4,4) or (8,8)
nsize = 4
zvals = np.resize(zvals,(nSize,nSize))

fig = pyplot.figure(2)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                           ['blue','green','yellow'],
                                           256)

img2 = pyplot.imshow(zvals,interpolation='nearest',
                    cmap = cmap2,
                    origin='lower')

pyplot.colorbar(img2,cmap=cmap2)
pyplot.yticks(range(nSize)[::-1])
pyplot.xticks(range(nSize))
pyplot.gca().invert_yaxis()

fig.savefig("image.png")
