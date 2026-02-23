class material:
    def __init__(self, name, rho, E, G, sigma_y, sigma_u):
        self.name = name #Material name
        self.rho = rho # Material volumic mass [kg.m^-3]
        self.E = E # Material Young's Modulus []
        self.G = G
        self.sigma_y = sigma_y
        self.sigma_u = sigma_u
        self.poisson = E/(2*G) - 1
