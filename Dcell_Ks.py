import numpy as np

class dcell_Ks:
    def __init__(self, a_over_b, data):
        self.a_over_b = a_over_b # Value of a/b for this serie - see reference plots
        self.data = data # Table of points extracted from the plot in the excel, ordered in increasing min(a,b)/sqrt(R*t): [[min(a,b)/sqrt(R*t),Ks]]

    def get_dcell_Ks(self, a, b, R, t):
        # Gives sigma_cr/sigma_0 for a given set of A_s, b, t
        # Inputs: Dcell panel width a [m] ; Dcell panel length b [m] ; Dcell panel radius R [m] ; Dcell panel thickness t [m]
        n = len(self.data) # Number of datapoints
        ratio = min(a,b)/np.sqrt(R*t) # Computing ratio used as abscyss
        value = 5 # Initializing with a low value to be conservative in case it fails
        if ratio >= self.data[n-1][0]: # If ratio is too high to fit on the graph, we take the last value
            value = self.data[n-1][1]
        else:
            k = 0
            while ratio > self.data[k][0]: # Finding first point that goes higher than the ratio we have
                k = k + 1
            if np.abs(ratio - self.data[k - 1][0]) < np.abs(ratio - self.data[k][0]): # Determining which point is closer, the one just before or just after our ratio
                value = self.data[k - 1][1]
            else:
                value = self.data[k][1]
        return value