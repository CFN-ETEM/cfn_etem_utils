from pyscope import tecnai
from gatansocket import GatanSocket
#from pyscope import dmsem
import time
from pyscope import gatan2
from PIL import Image

g = GatanSocket()
np_img = g.GetImage('gain normalized', 2672, 4008, 1, 0, 0, 2672, 4008, 3, 0)
pil_img = Image.fromarray(np_img, 'I;16')
pil_img.save('normal.tiff')
t = tecnai.Tecnai()
#print t.getBeamShift()
#print t.getRawImgeShift()
print t.getSpotSize()
print t.calculateAcquireParams()
#print t.getColumnPressure()


t= gatan2.Gatan()
#print t.getImage()

t.setExposureType('dark')
np_imgdark = t.getImage()
print t.getDimension()
print t.getOffset()
print t.getExposureType()
print t.getExposureTime()
print t.getBinning()
print np_imgdark.dtype, np_imgdark.shape
pil_imgdark = Image.fromarray(np_imgdark, 'I;16')

pil_imgdark.save('dark.tiff')

t.setExposureType('normal')
np_img = t.getImage()

pil_img = Image.fromarray(np_img, 'I;16')
pil_img.save('normal.tiff')
print np_img.dtype

np_imgsubed = np_img - np_imgdark
pil_img = Image.fromarray(np_imgsubed, 'I;16')
pil_img.save('subed.tiff')

#print t.__dict__.keys()
#print type(t)

#k = dmsem.GatanOrius()
#k.setExposureTime(200)
#print k.getImage()
#print type(k)
#print k.__dict__.keys()
#print k.getExposureType()
#print k.getBinning()

