from Layer import *
class FCNetwork:
    def __init__(self):
        self.layers = []
    def addLayer(self,neurons):
        self.layers.append(Layer(neurons))