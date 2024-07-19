import random

class Node:
    def __init__(self) -> None:
        self.weight = random.uniform(-1.0,1.0)
        self.bias = random.uniform(-1.0,1.0)
    def printNode(self) -> None:
        print(self.weight)
        #print("w=",self.weight," b=",self.bias)
    # def __init__(self) -> None:
    #     self.weight = 1
    #     self.bias = 0
    