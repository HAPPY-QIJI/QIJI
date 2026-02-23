import numpy as np

class fararserie:
    def __init__(self, value, data):
        self.value = value # Value of the farar coeff associated to this serie
        self.data = data # Table of points extracted from the plot in the excel: [[x0,y0],[x1,y1],...]

    def min_dist(self, point):
        min = 1e9
        closest_point = [0,0]
        for datapoint in self.data:
            dist = np.sqrt((point[0]-datapoint[0])**2 + (point[1]-datapoint[1])**2)
            if dist<min:
                min = dist
                closest_point = datapoint
        return (min, closest_point)