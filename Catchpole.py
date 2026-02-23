import numpy as np

class catchpole:
    def __init__(self, ts_over_t, data):
        self.ts_over_t = ts_over_t # Value of ts/t for this serie
        self.data = data # Table of points extracted from the plot in the excel, ordered in increasing A_s/bt: [[A_s/(b*t),sigma_cr/sigma_0]]

    def get_catchpole_coeff(self, A_s, b, t):
        # Gives sigma_cr/sigma_0 for a given set of A_s, b, t
        # Inputs: Stringer Area A_s [m²] ; Panel width b [m] ; Skin thickness t [m]
        n = len(self.data) # Number of datapoints
        ratio = A_s / b / t # Computing ratio used as abscyss
        value = 100 # Initializing with a ridiculous value
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