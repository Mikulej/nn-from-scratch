from Node import *
class Layer:
    def __init__(self,amount):
        self.nodes = []
        i = 0
        for i in range(amount):
            self.nodes.append(Node())
            Node.printNode(self.nodes[i])

    