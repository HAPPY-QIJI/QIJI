from Plane import plane
from Requirements import requirements
from Wing import wing
from WingSection import wingSection
from Materials import material
import numpy as np


# Plane Inputs
x_CoG = 12.13 #See Ola's LoadCase3 excel
x_CoP = 11.27 + 2.85/4 #See Project Brief
x_Emp = 28.896 #See Project Brief
MTOW = 28800 #See Project Brief
engine_mass = 400 #See Project Brief

# Load Case
maxLoadFactor = 3.75
minLoadFactor = -1.5
q = 1/2 * 0.5489 * 167.2**2 #Dynamic pressure
engine_thrust = 1811 #Single engine thrust [N] - From Load_Case1 excel on teams

#Number of wing sections
Nsections = 250

# Define Materials - Data to refine

# Aluminum alloys
al_2024_T3 = material(
    name="Aluminum 2024-T3",
    rho=2780,          # kg/m^3
    E=73.1e9,          # Pa
    G=28.0e9,          # Pa
    sigma_y=325e6,     # Pa
    sigma_u=470e6      # Pa
)

al_7075_T6 = material(
    name="Aluminum 7075-T6",
    rho=2810,          # kg/m^3
    E=71.7e9,          # Pa
    G=26.9e9,          # Pa
    sigma_y=505e6,     # Pa
    sigma_u=570e6      # Pa
)

# Titanium alloy (often used for highly loaded fittings / spars)
ti_6al_4v = material(
    name="Titanium Ti-6Al-4V",
    rho=4430,          # kg/m^3
    E=113.8e9,         # Pa
    G=44.0e9,          # Pa
    sigma_y=880e6,     # Pa
    sigma_u=950e6      # Pa
)

# Composite (simplified isotropic equivalent for preliminary sizing)
cfrp_ud_equiv = material(
    name="CFRP UD (isotropic equiv.)",
    rho=1600,          # kg/m^3
    E=135e9,           # Pa
    G=5.0e9,           # Pa (very approximate)
    sigma_y=600e6,     # Pa (pseudo-yield)
    sigma_u=1000e6     # Pa
)



# Wing Inputs
y_root = 1 #See Project Brief
span = 15.5 - y_root #See Project Brief
ratio_frontSpar = 0.15 #See Project Brief
ratio_rearSpar = 0.65 #See Project Brief
y_engines = [3.548, 6.768, 9.988] #See Project Brief
skin_material_table = [al_2024_T3 for k in range(Nsections)]
t_skin_table = [4e-3 for k in range(Nsections)]
spar_material = al_7075_T6
t_frontSpar_table = [1e-1 for k in range(Nsections)]
t_rearSpar_table = [1e-1 for k in range(Nsections)]
stringer_material = al_7075_T6
stringer_area = 3e-4 #m^-2
stringer_t_table = [4e-3 for k in range(Nsections)]
stringer_pitch_upper = 0.1 #m
stringer_pitch_lower = 0.1 #m
rib_stations = [1.01, 1.43617647,  1.86235294,  2.28852941,  2.71470588,  3.14088235,
  3.56705882,  3.99323529,  4.41941176,  4.84558824,  5.27176471,  5.69794118,
  6.12411765,  6.55029412,  6.97647059,  7.40264706,  7.82882353,  8.255,
 11.23823529, 11.66441176, 12.09058824, 12.51676471, 12.94294118, 13.36911765,
 13.79529412, 14.22147059, 14.64764706, 15.07382353, 15.5] #m
rib_t_table = [0.05 for k in range(len(rib_stations))] #m
rib_material = al_7075_T6
Cm = 0.05 # Aero Moment coefficient
        





# Creation of Plane
Plane = plane(x_CoG, x_CoP, x_Emp, MTOW, engine_mass, engine_thrust)
# Creation of requirements
Req = requirements(maxLoadFactor, minLoadFactor)
Req.setHalfLift(Plane)
# Creation of wing
Wing = wing(span, y_root, y_engines, Nsections, skin_material_table, t_skin_table_lower, t_skin_table_upper, spar_material, ratio_frontSpar, t_frontSpar_table, ratio_rearSpar, t_rearSpar_table, stringer_material, stringer_area, stringer_t_table, stringer_pitch_upper, stringer_pitch_lower, rib_stations, rib_t_table, rib_material, t_DCell_skin_table, DCellrib_stations, DCellrib_t)
# Setting up sections
Wing.setSections()

#Plotting loads
Wing.plot_loads(maxLoadFactor, Req.maxHalfLift, q, Cm, Plane.engine_mass, engine_thrust)

#Wing Mass
Wing_mass = Wing.wingMass()
print("Wing Mass: ",Wing_mass,"kg")
