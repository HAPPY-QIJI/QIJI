import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Plane import plane
from Requirements import requirements
from Wing import wing
from WingSection import wingSection
from Materials import material
from farar import fararserie
from Catchpole import catchpole
from Dcell_Ks import dcell_Ks
from Dataseries import dataseries


# Define Materials

# Aluminum alloys
al_2024_T3 = material(
    name="Aluminum 2024-T3",
    rho=2770,          # kg/m^3
    E=72e9,          # Pa
    G=28.0e9,          # Pa
    sigma_y=325e6,     # Pa
    sigma_u=470e6      # Pa
)

al_7075_T6 = material(
    name="Aluminum 7075-T6",
    rho=2800,          # kg/m^3
    E=71e9,          # Pa
    G=26.9e9,          # Pa
    sigma_y=505e6,     # Pa
    sigma_u=570e6      # Pa
)

# Titanium alloy
ti_6al_4v = material(
    name="Titanium Ti-6Al-4V",
    rho=4420,          # kg/m^3
    E=110e9,         # Pa
    G=42e9,          # Pa
    sigma_y=880e6,     # Pa
    sigma_u=950e6      # Pa
)


# Getting local compression buckling K data
df = pd.read_csv("Data/Local Buckling Data/All_data.csv",
                 sep=";", decimal=",")
ids = df.iloc[:,0].unique()
local_buckling_series = [
    df[df.iloc[:,0] == sid].iloc[:,1:3].values.tolist()
    for sid in ids
]
local_buckling = dataseries(local_buckling_series, ids)

# Getting panel compression buckling K data
panel_c_buckling_raw_data = pd.read_csv("Data\Panel compression data.csv", sep=";", decimal=",")
panel_c_buckling_data = [[]]
for k in range(len(panel_c_buckling_raw_data)):
    panel_c_buckling_data[0].append([panel_c_buckling_raw_data.iloc[k,0],panel_c_buckling_raw_data.iloc[k,1]])
panel_c_buckling = dataseries(panel_c_buckling_data)

# Getting panel shear buckling data
panel_s_buckling_raw_data = pd.read_csv("Data\Panel shear.csv", sep=";", decimal=",")
panel_s_buckling_data = [[]]
for k in range(len(panel_s_buckling_raw_data)):
    panel_s_buckling_data[0].append([panel_s_buckling_raw_data.iloc[k,0],panel_s_buckling_raw_data.iloc[k,1]])
panel_s_buckling = dataseries(panel_s_buckling_data)

# Getting Dcell compression buckling data
dcell_c_buckling_raw_data = pd.read_csv("Data\Dcell compression.csv", sep=";", decimal=",")
dcell_c_buckling_data = [[]]
for k in range(len(dcell_c_buckling_raw_data)):
    dcell_c_buckling_data[0].append([dcell_c_buckling_raw_data.iloc[k,0],dcell_c_buckling_raw_data.iloc[k,1]])
dcell_c_buckling = dataseries(dcell_c_buckling_data)

# Getting Dcell shear buckling data
df = pd.read_csv("Data\Dcell shear.csv", sep=";", decimal=",")
ids = df.iloc[:,0].unique()
dcell_s_buckling_series = [
    df[df.iloc[:,0] == sid].iloc[:,1:3].values.tolist()
    for sid in ids
]
dcell_s_buckling = dataseries(dcell_s_buckling_series, ids)

# Creation of farar data
f_50 = fararserie(0.5, [[0.4346880496246004,1.804306220095694], [0.3917960549955557,1.597607655502392], [0.3409043629555383,1.395215311004785], [0.2977286911133384,1.201435406698565], [0.2625527166821113,1.003349282296651], [0.2194716039109631,0.8052631578947369], [0.1762959320687633,0.6114832535885166], [0.1413090757796395,0.4047846889952154], [0.1273899805208314,0.3186602870813398], [0.2231026722393479,0.2799043062200957], [0.3655842805000283,0.2712918660287083], [0.412958375096923,1.713875598086124], [0.3621612421279573,1.507177033492823], [0.3192692474989125,1.300478468899521], [0.283904154925582,1.111004784688995], [0.2330124628855647,0.9086124401913874], [0.2057416267942583,0.7105263157894737], [0.1626605140231102,0.5124401913875596]])
f_60 = fararserie(0.6, [[0.6323165081226243,1.804306220095694], [0.5736142368137375,1.597607655502392], [0.4990071297539571,1.395215311004785], [0.4321160428919947,1.201435406698565], [0.3654140741721353,0.999043062200957], [0.3064281256500936,0.8052631578947369], [0.2475367361991035,0.607177033492823], [0.1966450441590862,0.4047846889952154], [0.2216842861735726,0.3444976076555024], [0.3566389923785388,0.3186602870813398], [0.4832157648883257,0.3143540669856459], [0.6029653724681807,1.700956937799043], [0.5363579628193731,1.494258373205742], [0.4694668759574107,1.300478468899521], [0.4026703481664996,1.102392344497608], [0.3359683794466403,0.8999999999999999], [0.2769824309245986,0.7062200956937799], [0.2180910414736085,0.508133971291866]])
f_70 = fararserie(0.7, [[1.011857707509881,1.8], [0.9349811827448606,1.700956937799043], [0.850294078710971,1.597607655502392], [0.7887550352705335,1.520095693779904], [0.712445864931823,1.395215311004785], [0.6514741759176959,1.291866028708134], [0.6059345272992038,1.205741626794258], [0.5527734175539459,1.106698564593301], [0.4997068668797396,1.003349282296651], [0.4544508954744029,0.9043062200956937], [0.4171946214800385,0.8009569377990429], [0.37184409100365,0.7062200956937799], [0.326588119598313,0.607177033492823], [0.2890481683907937,0.5167464114832536], [0.2516973353253777,0.4177033492822966], [0.3395994477750249,0.3746411483253589], [0.4823647332488605,0.3531100478468898], [0.6089415057586474,0.3488038277511962], [0.7588554570039905,0.3617224880382774], [0.8693382756207804,0.3703349282296649]])
f_75 = fararserie(0.75, [[1.24901185770751,1.8], [1.156230497191596,1.705263157894737], [1.055827675548915,1.597607655502392], [0.9630463150330011,1.502870813397129], [0.878359210999111,1.399521531100478], [0.8014826862340902,1.300478468899521], [0.7325112998089907,1.201435406698565], [0.6793501900637329,1.102392344497608], [0.6103788036386331,1.003349282296651], [0.5572176938933753,0.9043062200956937], [0.5041511432191689,0.8009569377990429], [0.4507063562607561,0.7148325358851675], [0.4056395029975225,0.607177033492823], [0.3678158745768481,0.5296650717703348], [0.3298031280140703,0.4607655502392345], [0.4332318399304045,0.430622009569378], [0.5677137507801123,0.426315789473684], [0.7336270968474006,0.430622009569378], [0.8914461864326646,0.4435406698564592], [1.072697013824536,0.4693779904306219], [1.198706431908013,0.490909090909091], [1.348336705940201,0.5167464114832536]])
f_80 = fararserie(0.8, [[1.675889328063241,1.8], [1.496529682092403,1.688038277511962], [1.364128070805832,1.597607655502392], [1.216105300981523,1.498564593301436], [1.083987366908107,1.395215311004785], [0.9754902887834029,1.296172248803828], [0.8906140666074096,1.201435406698565], [0.7980218242335988,1.098086124401914], [0.7211452994685781,0.999043062200957], [0.6598899332412957,0.9086124401913874], [0.5911076649582994,0.8009569377990429], [0.529946857802069,0.7062200956937799], [0.4688806097168901,0.607177033492823], [0.4235300792405015,0.5124401913875596], [0.5582011082323126,0.4995215311004784], [0.7400192900504945,0.4995215311004784], [0.8977438205647066,0.5167464114832536], [1.055279232936816,0.5425837320574161], [1.196909809558031,0.5727272727272725], [1.330540688768273,0.607177033492823], [1.464077008907464,0.6459330143540667], [1.597613329046655,0.684688995215311], [1.738865669383664,0.7320574162679425], [1.872401989522855,0.7708133971291866], [1.998033171322125,0.8095693779904307]])
f_85 = fararserie(0.85, [[1.971499895985022,1.657894736842105], [1.830531232861168,1.597607655502392], [1.67413052934168,1.520095693779904], [1.502108667284453,1.433971291866029], [1.345707963764964,1.356459330143541], [1.189685496529682,1.261722488038278], [1.073188720994005,1.166985645933014], [0.9722185449250147,1.085167464114833], [0.8951529020178906,0.9947368421052631], [0.8022769824309246,0.9043062200956937], [0.7175898783970345,0.8009569377990429], [0.6484293738298317,0.7105263157894737], [0.5874576848157043,0.607177033492823], [0.7061293189855702,0.6028708133971292], [0.8715698696976002,0.6287081339712917], [0.997484728710025,0.6545454545454545], [1.138926187189137,0.6933014354066986], [1.256746789719538,0.7277511961722487], [1.366662253910017,0.762200956937799], [1.476577718100497,0.7966507177033493], [1.586493182290977,0.8311004784688996], [1.688598067212588,0.861244019138756], [1.790513833992095,0.8999999999999999], [1.908239877451444,0.9387559808612439], [1.994629044764264,0.9645933014354067]])
f_90 = fararserie(0.9, [[1.976795203963916,1.416746411483254], [1.851164022164646,1.37799043062201], [1.709911681827638,1.330622009569378], [1.615995612459103,1.28755980861244], [1.498458687141857,1.240191387559809], [1.396921156646557,1.184210526315789], [1.311004784688995,1.136842105263158], [1.201845793066929,1.067942583732057], [1.100686498855835,0.9947368421052631], [1.022958942451349,0.9344497607655502], [0.9457041814021221,0.8526315789473684], [0.8836923426064263,0.7966507177033493], [0.829774760292755,0.7320574162679425], [0.9713107778429184,0.7665071770334928], [1.104941657053161,0.8009569377990429], [1.238477977192352,0.839712918660287], [1.372014297331543,0.8784688995215311], [1.497456360988709,0.9258373205741627], [1.63880326039677,0.9688995215311005], [1.772150462393858,1.016267942583732], [1.889876505853207,1.055023923444976], [1.991886831703765,1.089473684210526]])
f_95 = fararserie(0.95, [[1.604345934905535,1.098086124401914], [1.533577926130453,1.080861244019139], [1.471093291979499,1.046411483253588], [1.432229513777256,1.016267942583732], [1.495092384212417,1.033492822966507], [1.557671577434423,1.063636363636363]])

farar_table = [f_50, f_60, f_70, f_75, f_80, f_85, f_90, f_95]

def rib_segment(a, b, spacing):
    # generate only interior ribs (exclude a and b)
    return np.arange(a + spacing, b, spacing)

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
engine_thrust = 30620 # See my excel estimate for LC1

# Creation of Plane
Plane = plane(x_CoG, x_CoP, x_Emp, MTOW, engine_mass, engine_thrust)

# Creation of requirements
Req = requirements(maxLoadFactor, minLoadFactor)
Req.setHalfLift(Plane)

# Wing Inputs
y_root = 1 #See Project Brief
span = 15.5 - y_root #See Project Brief
ratio_frontSpar = 0.15 #See Project Brief
ratio_rearSpar = 0.65 #See Project Brief
y_engines = [3.548, 6.768, 9.988] #See Project Brief
Cm = 0.05 # Aero Moment coefficient - Conservative

# Design inputs
# Initialization
Nsections = 200
skin_material = al_2024_T3
DCell_rib_t = 0.0035 #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_frontSpar_web_table = [0.005 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_rearSpar_web_table = [0.004 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_Spar_cap_table = [0.011 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_dcell_lower_table = [0.007 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_dcell_mid_table = [0.007 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
t_dcell_upper_table = [0.007 for k in range(Nsections)] #m, ONLY USED TO INITIALIZE, SHOULD JUST BE CLOSE ENOUGH TO REAL VALUE
engine_rib_t = 0.008 #m

# Ranges of parameters
range_spar_material = [al_7075_T6, ti_6al_4v]
range_stringer_material = [al_2024_T3, al_7075_T6]
range_rib_material = [al_7075_T6, al_2024_T3]
range_t_skin_zones_lower = [
    [(0.2, 0.008+i*0.001), (0.7, 0.005+j*0.001), (1.0, 0.002+k*0.001)]
    for i in range(3)   # root
    for j in range(3)   # mid
    for k in range(1)   # tip
]
range_t_skin_zones_upper = [
    [(0.2, 0.008+i*0.001), (0.7, 0.005+j*0.001), (1.0, 0.002+k*0.001)]
    for i in range(3)   # root
    for j in range(3)   # mid
    for k in range(1)   # tip
]

range_stringer_area = [0.0003 + i*0.00015 for i in range(2)]

range_stringer_t_table = [
    [(0.003 + i*0.0015) for _ in range(Nsections)]
    for i in range(3)
]

range_stringer_pitch_upper = [0.10 + i*0.05 for i in range(3)]
range_stringer_pitch_lower = [0.10 + i*0.05 for i in range(3)]
range_rib_spacing_root_to_eng1 = [0.25, 0.3]
range_rib_spacing_eng1_to_eng2 = [0.3, 0.35, 0.45]
range_rib_spacing_eng2_to_eng3 = [0.55]
range_rib_spacing_eng3_to_tip  = [0.60]

def thickness_table_zones(rib_stations, zones): # Defining the skin ticikness tables based on the zones and rib stations
    """
    zones: list of (end_frac, thickness) in increasing end_frac, last end_frac=1.0
    """
    ribs = np.asarray(rib_stations, float)
    y0, y1 = ribs[0], ribs[-1]
    frac = (ribs - y0) / (y1 - y0 + 1e-12)

    out = np.empty_like(frac)
    start = 0.0
    for end, t in zones:
        mask = (frac >= start) & (frac <= end + 1e-12)
        out[mask] = t
        start = end
    return out.tolist()

def explore_design(panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling, test=False):
    results = []

    best = None  # store only what’s needed to rebuild & print later

    n = (len(range_t_skin_zones_lower)*len(range_t_skin_zones_upper)*len(range_stringer_area)*len(range_stringer_t_table)*
         len(range_stringer_pitch_upper)*len(range_stringer_pitch_lower)*len(range_rib_spacing_root_to_eng1)*
         len(range_rib_spacing_eng1_to_eng2)*len(range_rib_spacing_eng2_to_eng3)*len(range_rib_spacing_eng3_to_tip)
         *len(range_spar_material)*len(range_stringer_material)*len(range_rib_material))
    print("Total number of cases: ", n)

    k = 0
    upper_skin_failures = 0
    lower_skin_failures = 0
    dcell_failures = 0
    rib_failures = 0
    spar_failures = 0
    pruning_failures = 0
    for t_skin_zones_lower in range_t_skin_zones_lower:
        for t_skin_zones_upper in range_t_skin_zones_upper:
            for stringer_area in range_stringer_area:
                for stringer_t_table in range_stringer_t_table:
                    for stringer_pitch_upper in range_stringer_pitch_upper:
                        for stringer_pitch_lower in range_stringer_pitch_lower:
                            for spacing_root_to_eng1 in range_rib_spacing_root_to_eng1:
                                for spacing_eng1_to_eng2 in range_rib_spacing_eng1_to_eng2:
                                    for spacing_eng2_to_eng3 in range_rib_spacing_eng2_to_eng3:
                                        for spacing_eng3_to_tip in range_rib_spacing_eng3_to_tip:
                                            for spar_material in range_spar_material:
                                                for rib_material in range_rib_material:
                                                    for stringer_material in range_stringer_material:
                                                        rib_stations = np.concatenate([[y_root + 0.001], rib_segment(y_root + 0.001, y_engines[0], spacing_root_to_eng1),
                                                                                        [y_engines[0]], rib_segment(y_engines[0], y_engines[1], spacing_eng1_to_eng2),
                                                                                        [y_engines[1]], rib_segment(y_engines[1], y_engines[2], spacing_eng2_to_eng3),
                                                                                        [y_engines[2]], rib_segment(y_engines[2], 15.5, spacing_eng3_to_tip)])
                                                        rib_t_table = [0.006 for _ in range(len(rib_stations))]
                                                        k += 1
                                                        t_skin_table_lower = thickness_table_zones(rib_stations, t_skin_zones_lower)
                                                        t_skin_table_upper = thickness_table_zones(rib_stations, t_skin_zones_upper)
                                                        Wing = wing(
                                                            span, y_root, y_engines, Nsections,
                                                            skin_material, t_skin_table_lower, t_skin_table_upper,
                                                            spar_material, ratio_frontSpar, t_frontSpar_web_table,
                                                            ratio_rearSpar, t_rearSpar_web_table, t_Spar_cap_table,
                                                            stringer_material, stringer_area, stringer_t_table,
                                                            stringer_pitch_upper, stringer_pitch_lower,
                                                            rib_stations, rib_t_table, engine_rib_t, rib_material,
                                                            t_dcell_lower_table, t_dcell_mid_table,
                                                            t_dcell_upper_table, DCell_rib_t
                                                        )

                                                        Wing.setSections()

                                                        if not Wing.pruning_check(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift, 
                                                            engine_mass, engine_thrust, q, Cm,
                                                            panel_c_buckling, local_buckling, panel_s_buckling
                                                            ):
                                                            pruning_failures+=1
                                                            if k%100 == 0:
                                                                print(k, '/', n, " :")
                                                                print('Pruning failures: ', pruning_failures,'/',k)
                                                                print('Lower skin failures: ', lower_skin_failures,'/',k)
                                                                print('Upper skin failures: ', upper_skin_failures,'/',k)
                                                                print('Dcell failures: ', dcell_failures,'/',k)
                                                                print('Rib failures: ', rib_failures,'/',k)
                                                                print('Spar failures: ', spar_failures,'/',k)

                                                            if test and k==1:
                                                                print('End of test, only one wing tested.')
                                                                print("Wing mass: ", wing_mass," kg")
                                                                return [],Wing
                                                            continue

                                                        Wing.optimize_rib_thicknesses(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift, 
                                                            engine_mass, engine_thrust, q, Cm,
                                                            panel_s_buckling
                                                        )

                                                        Wing.optimizeDcell(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift,
                                                            engine_mass, engine_thrust, q, Cm,
                                                            dcell_c_buckling, dcell_s_buckling
                                                        )


                                                        Wing.optimizeSparThickness(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift,
                                                            engine_mass, engine_thrust, q, Cm,
                                                            panel_s_buckling
                                                        )

                                                        Wing.optimize_rib_thicknesses(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift, 
                                                            engine_mass, engine_thrust, q, Cm,
                                                            panel_s_buckling
                                                        )

                                                        Wing.optimizeDcell(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift,
                                                            engine_mass, engine_thrust, q, Cm,
                                                            dcell_c_buckling, dcell_s_buckling
                                                        )

                                                        Wing.optimizeSparThickness(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift,
                                                            engine_mass, engine_thrust, q, Cm,
                                                            panel_s_buckling
                                                        )

                                                        lower_pass, upper_pass, spar_pass, dcell_pass, rib_pass = Wing.check_wing(
                                                            Req.maxLoadFactor, Req.maxHalfLift,
                                                            Req.minLoadFactor, Req.minHalfLift,
                                                            engine_mass, engine_thrust,
                                                            q, Cm, panel_c_buckling, local_buckling,
                                                            panel_s_buckling, dcell_c_buckling,
                                                            dcell_s_buckling
                                                        )

                                                        wing_mass = Wing.wingMass()

                                                        # collect what plotting needs
                                                        rec = {
                                                            "mass": wing_mass,
                                                            "upper_fail": Wing.failed_upper_sections[0],
                                                            "lower_fail": Wing.failed_lower_sections[0],
                                                            "spar_fail":  Wing.failed_spars[0],
                                                            "dcell_fail": Wing.failed_dcells[0],
                                                            "rib_fail": Wing.failed_ribs[0],
                                                            "upper_pass": upper_pass,
                                                            "lower_pass": lower_pass,
                                                            "spar_pass":  spar_pass,
                                                            "dcell_pass": dcell_pass,
                                                            "rib_pass": rib_pass,
                                                        }
                                                        results.append(rec)

                                                        # best design tracking (store params to rebuild)
                                                        if upper_pass and lower_pass and spar_pass and dcell_pass and rib_pass:
                                                            if (best is None) or (wing_mass < best["mass"]):
                                                                best = {
                                                                    "mass": wing_mass,
                                                                    "params": {
                                                                        "t_skin_table_lower": t_skin_table_lower,
                                                                        "t_skin_table_upper": t_skin_table_upper,
                                                                        "stringer_material": stringer_material,
                                                                        "stringer_area": stringer_area,
                                                                        "stringer_t_table": stringer_t_table,
                                                                        "stringer_pitch_upper": stringer_pitch_upper,
                                                                        "stringer_pitch_lower": stringer_pitch_lower,
                                                                        "rib_material": rib_material,
                                                                        "rib_stations": rib_stations,
                                                                        "rib_t_table": rib_t_table,
                                                                        "spar_material": spar_material,
                                                                        "engine_rib_t": engine_rib_t,
                                                                    }
                                                                }

                                                        if not lower_pass:
                                                            lower_skin_failures+=1
                                                        if not upper_pass:
                                                            upper_skin_failures+=1
                                                        if not dcell_pass:
                                                            dcell_failures+=1
                                                        if not spar_pass:
                                                            spar_failures+=1
                                                        if not rib_pass:
                                                            rib_failures+=1
                                                        if k%100 == 0:
                                                            print(k, '/', n, " :")
                                                            print('Pruning failures: ', pruning_failures,'/',k)
                                                            print('Lower skin failures: ', lower_skin_failures,'/',k)
                                                            print('Upper skin failures: ', upper_skin_failures,'/',k)
                                                            print('Dcell failures: ', dcell_failures,'/',k)
                                                            print('Rib failures: ', rib_failures,'/',k)
                                                            print('Spar failures: ', spar_failures,'/',k)

                                                        if test and k==1:
                                                            print('End of test, only one wing tested.')
                                                            print("Wing mass: ", wing_mass," kg")
                                                            return [],Wing
    return results, best



def plot_design(panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling, test=False):
    data, best = explore_design(panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling, test)
    if test:
        return

    x_data = []
    y_data_upper = []
    y_data_lower = []
    y_data_spars = []
    y_data_dcells = []
    y_data_ribs = []
    colors_upper = []
    colors_lower = []
    colors_spars = []
    colors_dcells = []
    colors_ribs = []

    for rec in data:
        wing_mass   = rec["mass"]
        upper_pass  = rec["upper_pass"]
        lower_pass  = rec["lower_pass"]
        spars_pass  = rec["spar_pass"]
        dcells_pass = rec["dcell_pass"]
        ribs_pass = rec["rib_pass"]

        x_data.append(wing_mass)
        y_data_upper.append(rec["upper_fail"])
        y_data_lower.append(rec["lower_fail"])
        y_data_spars.append(rec["spar_fail"])
        y_data_dcells.append(rec["dcell_fail"])
        y_data_ribs.append(rec["rib_fail"])

        # colour logic
        colors_upper.append("green" if upper_pass else ("red" if not (lower_pass and spars_pass and dcells_pass and ribs_pass) else "orange"))
        colors_lower.append("green" if lower_pass else ("red" if not (upper_pass and spars_pass and dcells_pass and ribs_pass) else "orange"))
        colors_spars.append("green" if spars_pass else ("red" if not (upper_pass and lower_pass and dcells_pass and ribs_pass) else "orange"))
        colors_dcells.append("green" if dcells_pass else ("red" if not (upper_pass and lower_pass and spars_pass and ribs_pass) else "orange"))
        colors_ribs.append("green" if ribs_pass else ("red" if not (upper_pass and lower_pass and spars_pass and dcells_pass) else "orange"))

    # Rebuild and print best wing
    if best is not None:
        p = best["params"]
        print("Done. Best wing mass:", best["mass"], "kg")

        best_wing = wing(
            span, y_root, y_engines, Nsections,
            skin_material, p["t_skin_table_lower"], p["t_skin_table_upper"],
            p["spar_material"], ratio_frontSpar, t_frontSpar_web_table,
            ratio_rearSpar, t_rearSpar_web_table, t_Spar_cap_table,
            p["stringer_material"], p["stringer_area"], p["stringer_t_table"],
            p["stringer_pitch_upper"], p["stringer_pitch_lower"],
            p["rib_stations"], p["rib_t_table"], p["engine_rib_t"], p["rib_material"],
            t_dcell_lower_table, t_dcell_mid_table, t_dcell_upper_table, DCell_rib_t
        )
        best_wing.setSections()
        best_wing.optimizeSparThickness(Req.maxLoadFactor, Req.maxHalfLift,
                                        Req.minLoadFactor, Req.minHalfLift, engine_mass, engine_thrust, q, Cm, panel_s_buckling)
        best_wing.optimize_rib_thicknesses(Req.maxLoadFactor, Req.maxHalfLift, Req.minLoadFactor, Req.minHalfLift,
                                           engine_mass, engine_thrust, q, Cm, panel_s_buckling)
        best_wing.check_wing(Req.maxLoadFactor, Req.maxHalfLift, Req.minLoadFactor, Req.minHalfLift,
                             engine_mass, engine_thrust, q, Cm, panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling)
        best_wing.display_characteristics()
    else:
        print("All designs failed. Please change structural characteristics.")

    # plotting failed sections
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(9, 11))
    axs[0].scatter(x_data, y_data_upper, s=5, c=colors_upper); axs[0].set_ylabel("Upper skin failures"); axs[0].grid(True)
    axs[1].scatter(x_data, y_data_lower, s=5, c=colors_lower); axs[1].set_ylabel("Lower skin failures"); axs[1].grid(True)
    axs[2].scatter(x_data, y_data_spars, s=5, c=colors_spars); axs[2].set_ylabel("Spar failures"); axs[2].grid(True)
    axs[3].scatter(x_data, y_data_dcells, s=5, c=colors_dcells); axs[3].set_ylabel("DCells failures"); axs[3].grid(True)
    axs[4].scatter(x_data, y_data_ribs, s=5, c=colors_ribs); axs[4].set_ylabel("Ribs failures"); axs[4].set_xlabel("Wing mass [kg]"); axs[4].grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting Farar coefficients
    best_wing.plot_farar(farar_table)

    # Printing mass split
    best_wing.mass_split()

    return best_wing




result = plot_design(panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling, test=False)