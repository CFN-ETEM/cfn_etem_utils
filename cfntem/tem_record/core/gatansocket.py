from numpy.random import randint, rand
import time

class GatanSocket:
    def __init__(self):
        pass
    
    def GetImage(self, *args):
        time.sleep(rand()*2)
        img = randint(0, 255*255, (2672, 4008), dtype='uint16')
        return img
