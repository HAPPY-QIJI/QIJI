import numpy as np

class dataseries:
    def __init__(self, data, serie_ids=None):
        self.data = data
        self.serie_ids = serie_ids

    def find_y(self, x, serie_id=None):

        # Select correct series
        if self.serie_ids is None or serie_id is None:
            best_id_index = 0
        else:
            distances = np.abs(np.array(self.serie_ids) - serie_id)
            best_id_index = np.argmin(distances)

        serie = self.data[best_id_index]

        xs = [p[0] for p in serie]
        ys = [p[1] for p in serie]

        # Clamp edges
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]

        # Find closest point
        for i in range(1, len(xs)):
            if x <= xs[i]:
                if abs(x - xs[i-1]) < abs(x - xs[i]):
                    return ys[i-1]
                else:
                    return ys[i]
