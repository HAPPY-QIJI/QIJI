class plane:
    def __init__(self, x_CoG, x_CoP, x_Emp, MTOW, engine_mass, engine_thrust):
        self.x_CoG = x_CoG #X station of center of gravity [m]
        self.x_CoP = x_CoP #X station of center of pressure [m]
        self.x_Emp = x_Emp #X station of empennage center of pressure [m]
        self.MTOW = MTOW #MTOW [kg]
        self.engine_mass = engine_mass #Engine mass [kg]
        self.engine_thrust = engine_thrust #Single engine thrust [N]