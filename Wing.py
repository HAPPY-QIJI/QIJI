import numpy as np
import matplotlib.pyplot as plt
from WingSection import wingSection

class wing:
    def __init__(self, span, y_root, y_engines, Nsections, skin_material, t_skin_table_lower, t_skin_table_upper, spar_material, ratio_frontSpar, t_frontSpar_web_table, ratio_rearSpar, t_rearSpar_web_table, t_Spar_cap_table, stringer_material, stringer_area, stringer_t_table, stringer_pitch_upper, stringer_pitch_lower, rib_stations, rib_t_table, engine_rib_t, rib_material, t_dcell_lower_table, t_dcell_mid_table, t_dcell_upper_table, DCell_rib_t):
        self.span = span #Span of one side of the wings [m]
        self.y_root = y_root #Y station of the root [m]
        self.y_engines = y_engines # Table containing the y stations of the engines [m]
        self.Nsections = Nsections #Number of sections for one side of the wings
        self.sections = [] #List of sections
        self.skin_material = skin_material #Table of materials of length Nsections
        self.t_skin_table_lower = t_skin_table_lower #Table of lower skin thicknesses ([m]) of the same length as rib_stations
        self.t_skin_table_upper = t_skin_table_upper #Table of upper skin thicknesses ([m]) of the same length as rib_stations
        self.spar_material = spar_material #Spar material
        self.ratio_frontSpar = ratio_frontSpar # Front spar X station as a ratio of chord
        self.t_frontSpar_web_table = t_frontSpar_web_table #Table of front spar thicknesses ([m]) of length Nsections
        self.ratio_rearSpar = ratio_rearSpar # Rear spar X station as a ratio of chord
        self.t_rearSpar_web_table = t_rearSpar_web_table #Table of rear spar thicknesses ([m]) of length Nsections
        self.t_Spar_cap_table = t_Spar_cap_table #Table of spar cap thicknesses ([m]) of length Nsections
        self.stringer_material = stringer_material # Stringer material
        self.stringer_area = stringer_area # Stringer Area [m²], assumed constant throughout the wing
        self.stringer_t_table = stringer_t_table # Table of stringer thicknesses ([m]) of length Nsections
        self.stringer_pitch_upper = stringer_pitch_upper # Stringer Pitch (ie distance between 2 stringers) [m], assumed constant throughout the wing
        self.stringer_pitch_lower = stringer_pitch_lower # Stringer Pitch (ie distance between 2 stringers) [m], assumed constant throughout the wing
        self.rib_stations = rib_stations # Table containing the Y stations of the ribs (same stations for skin panels and Dcell), measured from the symmetry plane [m]
        self.rib_t_table = rib_t_table # Table containing the thicknesses of the ribs, must be of the same length as self.rib_stations [m]
        self.engine_rib_t = engine_rib_t
        self.rib_material = rib_material # Material of ribs AND DCELL RIBS, assumed same for all ribs
        self.DCell_rib_t = DCell_rib_t # Thickness of the DCell ribs [m]
        self.t_dcell_lower_table = t_dcell_lower_table # Thicknesses of the Dcell lower skin, ONLY USED FOR INITIAL LOAD CALCULATIONS, LATER OPTIMIZED [m]
        self.t_dcell_mid_table = t_dcell_mid_table # Thicknesses of the Dcell mid skin, ONLY USED FOR INITIAL LOAD CALCULATIONS, LATER OPTIMIZED [m]
        self.t_dcell_upper_table = t_dcell_upper_table # Thicknesses of the Dcell upper skin, ONLY USED FOR INITIAL LOAD CALCULATIONS, LATER OPTIMIZED [m]
        self.failed_upper_sections = None # Table of failed upper sections, self.failed_upper_sections[0] contains the number of failed sections and self.failed_upper_sections[1] a list of failed Y stations
        self.failed_lower_sections = None # Table of failed lower sections, self.failed_lower_sections[0] contains the number of failed sections and self.failed_lower_sections[1] a list of failed Y stations
        self.failed_spars = None # Table of failed spars, self.failed_spars[0] contains the number of failed spars and self.failed_spars[1] a list of failed Y stations
        self.failed_dcells = None # Table of failed dcells, self.failed_dcells[0] contains the number of failed dcells and self.failed_dcells[1] a list of failed Y stations
        self.failed_ribs = None # Table of failed ribs, self.failed_ribs[0] contains the number of failed ribs and self.failed_ribs[1] a list of failed Y stations

    def local_spacing(self, y, rib_positions):
        '''
        Auxiliary function for rib spacing in set_sections
        Inputs:
        y station
        table of y stations of the ribs
        Returns value of rib spacing for the given y station
        '''
        # rib_positions: sorted 1D array/list of rib y-stations
        rib_positions = np.asarray(rib_positions, dtype=float)
        rib_positions.sort()

        # clamp: if y outside the rib range, use nearest segment
        if y <= rib_positions[0]:
            return rib_positions[1] - rib_positions[0]
        if y >= rib_positions[-1]:
            return rib_positions[-1] - rib_positions[-2]

        i = np.searchsorted(rib_positions, y, side="right")
        return rib_positions[i] - rib_positions[i-1]

    def skin_t_at_y(self, y):
        '''
        Auxiliary function to determine skin thickness at a given y station
        Inputs: y station
        Return the lower and upper skin thicknesses at this y station
        '''
        if y < self.rib_stations[0]:
            index = 0
        elif y > self.rib_stations[-1]:
            index = len(self.rib_stations) - 1
        else:
            index = 0
            while index < len(self.rib_stations) - 1 and y >= self.rib_stations[index]:
                index = index + 1
        t_lower = self.t_skin_table_lower[index]
        t_upper = self.t_skin_table_upper[index]
        return (t_lower, t_upper)

    def setSections(self):
        '''
        Initialising sections in the self.sections table
        Uses the spar thicknesses, rib thicknesses, and dcell skin thicknesses provided when the wing object was created,
        but these values will be later modified by optimizeSparThickness, optimize_DCell, and optimize_rib_thicknesses
        '''
        n_engine = 0
        for k in range(self.Nsections):
            engine_flag = False
            y = 15.5 - self.span + k * self.span/(self.Nsections-1) # Allows to change span while still having the tip @ Y=15.5
            # Chord
            if y < 3.548: #Rectangular part of the wing
                chord = 2.85
            else: # Tapered part of the wing
                chord = 3.25 - 0.113*y
            # Presence of engine
            if n_engine < len(self.y_engines):
                if np.abs(self.y_engines[n_engine]-y) < self.span/(2*(self.Nsections-1)):
                    engine_flag = True
                    n_engine = n_engine + 1
            # Skin thicknesses
            t_skin_lower, t_skin_upper = self.skin_t_at_y(y)
            # Stringers
            width = chord * (self.ratio_rearSpar - self.ratio_frontSpar)
            n_stringers_upper = max(0, int(np.ceil(width / self.stringer_pitch_upper)) - 1) # Number of stringers on upper part
            n_stringers_lower = max(0, int(np.ceil(width / self.stringer_pitch_lower)) - 1) # Number of stringers on lower part
            # Rib spacing (same for Dcell and skin panels)
            rib_spacing = self.local_spacing(y, self.rib_stations)
            # Presence of rib at the given section
            n_ribs = len(self.rib_stations)
            rib_thickness = 0
            dcell_rib_thickness = 0
            for i in range(n_ribs):
                if np.abs(self.rib_stations[i]-y) < self.span/(2*(self.Nsections-1)):
                    rib_thickness = self.rib_t_table[i]
                    dcell_rib_thickness = self.DCell_rib_t
                if engine_flag:
                    rib_thickness = self.engine_rib_t
                    dcell_rib_thickness = self.engine_rib_t
            # Initialization of section
            section = wingSection(y, chord, engine_flag, self.skin_material, t_skin_lower, t_skin_upper, self.spar_material, self.ratio_frontSpar, self.t_frontSpar_web_table[k], self.ratio_rearSpar, self.t_rearSpar_web_table[k], self.t_Spar_cap_table[k], 0.05*chord, self.stringer_area, self.stringer_t_table[k], self.stringer_material, n_stringers_upper, n_stringers_lower, rib_spacing, rib_thickness, self.t_dcell_lower_table[k], self.t_dcell_mid_table[k], self.t_dcell_upper_table[k], dcell_rib_thickness)
            self.sections.append(section)

    def wingMass(self):
        '''
        Computing mass of the wing
        Returns the mass of the wing in kg, including non-wing-box mass (estimated as 20% of the total wing mass)
        '''
        mass = 0
        for k in range(self.Nsections):
            section = self.sections[k]
            mass_rib = section.rib_thickness * section.chord * (section.ratio_rearSpar - section.ratio_frontSpar) * 0.1013*section.chord * self.rib_material.rho # Rib thickness is zero if there's no rib at this station
            dcell_area = np.pi/2 * section.chord*0.1013 * section.chord*self.ratio_frontSpar# DCell area approximated as a half ellipse
            mass_dcell_rib = section.dcell_rib_thickness * dcell_area * self.rib_material.rho
            mass_distributed = section.mass_per_span(self.span, self.Nsections) * self.span / (self.Nsections - 1)
            mass = mass + mass_distributed + mass_rib + mass_dcell_rib
        return mass
    
    def SF_BM(self, load_factor, lift, engine_mass):
        '''
        Computing Shear Force and Bending Moment distribution, from root to tip
        Inputs: load factor , single wing lift [N], engine mass [kg]
        Return lists of length Nsections containing the shear force and bending moment, from root to tip
        '''
        V = [0 for i in range(self.Nsections)]
        BM = [0 for i in range(self.Nsections)]
        sum_V = 0
        sum_BM = 0
        dy = self.span/(self.Nsections-1)
        for k in range(self.Nsections):
            i = self.Nsections - 1 - k
            V[i] = sum_V
            sum_V = sum_V + self.sections[i].load(load_factor, self.span,self.y_root, self.Nsections, lift, engine_mass, self.rib_material)
            sum_BM = sum_BM + V[i] * dy
            BM[i] = sum_BM
        return V, BM
    
    
    def torque(self, load_factor, lift, q, Cm, engine_mass, engine_thrust):
        '''
        Computing Torque distribution, from root to tip
        Inputs:
        load factor
        single wing lift [N]
        Dynamic pressure q [kg.m^-1.s^-2]
        Aero moment coefficient Cm
        Single engine mass [kg]
        Single engine thrust [N]
        Returns a list of length Nsections storing the torque from root to tip
        '''
        T = [0 for i in range(self.Nsections)]
        sum = 0
        for k in range(self.Nsections):
            T[self.Nsections - 1 - k] = sum
            increment = self.sections[self.Nsections - 1 - k].torque_section(load_factor, self.span, self.y_root, self.Nsections, q, Cm, lift, engine_mass, engine_thrust)
            sum = sum + increment
        return T
    
    def plot_loads(self, load_factor, lift, q, Cm, engine_mass, engine_thrust):
        '''
        Plotting SF, BM, and torque along the span
        Inputs:
        load factor
        single wing lift [N]
        Dynamic pressure q [kg.m^-1.s^-2]
        Aero moment coefficient Cm
        Single engine mass [kg]
        Single engine thrust [N]
        '''
        SF, BM = self.SF_BM(load_factor, lift, engine_mass)
        T = self.torque(load_factor, lift, q, Cm, engine_mass, engine_thrust)
        Y = [self.sections[k].y for k in range(self.Nsections) ]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

        # Shear force
        axs[0].plot(Y, SF)
        axs[0].set_ylabel("Shear Force [N]")
        axs[0].grid(True)

        # Bending moment
        axs[1].plot(Y, BM)
        axs[1].set_ylabel("Bending Moment [N·m]")
        axs[1].grid(True)

        # Torque
        axs[2].plot(Y, T)
        axs[2].set_ylabel("Torque [N·m]")
        axs[2].set_xlabel("Spanwise position y [m]")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def optimizeSparThickness(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm, panel_s_buckling, flag="variable"):
        '''
        Sets up tables of spar thicknesses and modifies section.t_spar for all sections
        Inputs:
        Maximum load factor considered
        Single wing lift for the maximum load factor considered [N]
        Minimum load factor considered
        Single wing lift for the minimum load factor considered [N]
        Single engine mass [kg]
        Single engine thrust [N]
        Dynamic pressure q [kg.m^-1.s^-2]
        Aero moment coefficient Cm
        Object of the Dataseries class containing data for shear buckling (ESDU 71005)       
        '''
        SF_maxload,BM_maxload = self.SF_BM(max_load_factor, max_lift, engine_mass)
        SF_minload,BM_minload = self.SF_BM(min_load_factor, min_lift, engine_mass)
        T_maxload = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        T_minload = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        t_front_web_current = 0
        t_rear_web_current = 0
        t_cap_current = 0
        dt = 1e-4  # All thicknesses will be rounded up to the next multiple of 0.1mm
        start_index = 0
        for k in range(self.Nsections):
            # Geometry
            section = self.sections[k]
            height = 0.1013*section.chord
            width = (self.ratio_rearSpar - self.ratio_frontSpar) * section.chord
            rib_spacing = section.rib_spacing
            chord = self.sections[k].chord
            spar_width = 0.05*chord
            stringer_spacing_upper = width/(section.n_stringers_upper+1)
            stringer_spacing_lower = width/(section.n_stringers_lower+1)
            t_eff_lower = section.t_skin_lower + self.stringer_area / stringer_spacing_lower
            t_eff_upper = section.t_skin_upper + self.stringer_area / stringer_spacing_upper
            # Web Material
            E_web = self.skin_material.E
            sigma_y_web = self.skin_material.sigma_y
            # Cap Material
            E_cap = self.spar_material.E
            sigma_y_cap = self.spar_material.sigma_y
            # Shear buckling coefficient for web
            x_ks = height / rib_spacing
            if x_ks > 1:
                x_ks = 1/x_ks
            Ks = panel_s_buckling.find_y(x_ks)
            # Loads
            SF_maxload_sec = SF_maxload[k]
            SF_minload_sec = SF_minload[k]
            BM_maxload_sec = BM_maxload[k]
            BM_minload_sec = BM_minload[k]
            T_maxload_sec = T_maxload[k]
            T_minload_sec = T_minload[k]

            # WEB SIZING - SHEAR
            # Shear buckling for n=n_max
            t_front_crit_s_buckling_max = ( np.abs(SF_maxload_sec + T_maxload_sec/width) * height / (2*Ks*E_web) )**(1/3)
            t_rear_crit_s_buckling_max = ( np.abs(SF_maxload_sec - T_maxload_sec/width) * height / (2*Ks*E_web) )**(1/3)
            # Shear buckling for n=n_min
            t_front_crit_s_buckling_min = ( np.abs(SF_minload_sec + T_minload_sec/width) * height / (2*Ks*E_web) )**(1/3)
            t_rear_crit_s_buckling_min = ( np.abs(SF_minload_sec - T_minload_sec/width) * height / (2*Ks*E_web) )**(1/3)
            # Shear yield for n=n_max
            t_front_crit_s_yield_max = np.sqrt(3) * np.abs(SF_maxload_sec + T_maxload_sec/width) / (2*height*sigma_y_web)
            t_rear_crit_s_yield_max = np.sqrt(3) * np.abs(SF_maxload_sec - T_maxload_sec/width) / (2*height*sigma_y_web)
            # Shear yield for n=n_min
            t_front_crit_s_yield_min = np.sqrt(3) * np.abs(SF_minload_sec + T_minload_sec/width) / (2*height*sigma_y_web)
            t_rear_crit_s_yield_min = np.sqrt(3) * np.abs(SF_minload_sec - T_minload_sec/width) / (2*height*sigma_y_web)
            # Keeping max web thickness
            t_web_front = max(t_front_crit_s_buckling_max, t_front_crit_s_buckling_min, t_front_crit_s_yield_max, t_front_crit_s_yield_min, 0.002)
            t_web_rear = max(t_rear_crit_s_buckling_max, t_rear_crit_s_buckling_min, t_rear_crit_s_yield_max, t_rear_crit_s_yield_min, 0.002)

            # CAP SIZING - AXIAL STRESS
            # Compressive buckling
            # Solving roots of polynome
            a = np.pi**2 * E_cap * 0.1 * chord * height
            b = (t_eff_lower+t_eff_upper)*width*height*np.pi**2 * E_cap / 2
            target_max = 24*rib_spacing**2 * np.abs(BM_maxload_sec)
            target_min = 24*rib_spacing**2 * np.abs(BM_minload_sec)
            target_abs = max(target_max, target_min)
            t_crit_c_buckling = 0.005
            current_poly = t_crit_c_buckling**2 * (a*t_crit_c_buckling + b)
            while current_poly < target_abs:
                t_crit_c_buckling += 0.0005
                current_poly = t_crit_c_buckling**2 * (a*t_crit_c_buckling + b)
            # Axial yield
            t_crit_c_yield_max = max(10/(chord*height) * (np.abs(BM_maxload_sec)/sigma_y_cap - width*height/2 * (t_eff_upper + t_eff_lower)),0.005)
            t_crit_c_yield_min = max(10/(chord*height) * (np.abs(BM_minload_sec)/sigma_y_cap - width*height/2 * (t_eff_upper + t_eff_lower)),0.005)
            # Keeping max cap thicknesses - Same for front and rear because compressive stress is the same
            t_cap = max(t_crit_c_buckling, t_crit_c_yield_max, t_crit_c_yield_min, 0.005)

            # UPDATING THICKNESSES
            if flag=="constant":
                # Spar web thicknesses
                if t_web_front > t_front_web_current:
                    t_front_web_current = t_web_front
                if t_web_rear > t_rear_web_current:
                    t_rear_web_current = t_web_rear
                # Spar cap thicknesses
                if t_cap > t_cap_current:
                    t_cap_current = t_cap
                # Checking if end of spar
                if section.rib_thickness > 0 or k == self.Nsections - 1:
                    end_index = k
                    t_front_web_rounded = max(np.ceil(t_front_web_current / dt) * dt, 0.002)
                    t_rear_web_rounded  = max(np.ceil(t_rear_web_current / dt) * dt, 0.002)
                    t_cap_rounded = max(np.ceil(t_cap_current / dt) * dt, 0.002)
                    for i in range(start_index, end_index + 1):
                        self.t_frontSpar_web_table[i] = t_front_web_rounded
                        self.t_rearSpar_web_table[i]  = t_rear_web_rounded
                        self.t_Spar_cap_table[i] = t_cap_rounded
                        self.sections[i].t_frontSpar_web = t_front_web_rounded
                        self.sections[i].t_rearSpar_web = t_rear_web_rounded
                        self.sections[i].t_Spar_cap = t_cap_rounded
                        self.sections[i].spar_cap_width = spar_width
                    t_front_web_current = 0
                    t_rear_web_current = 0
                    t_cap_current = 0
                    start_index = k+1
            elif flag=="variable":
                t_front_web_rounded = max(np.ceil(t_web_front / dt) * dt, 0.002)
                t_rear_web_rounded  = max(np.ceil(t_web_rear / dt) * dt, 0.002)
                t_cap_rounded = max(np.ceil(t_cap / dt) * dt, 0.002)
                self.t_frontSpar_web_table[k] = t_front_web_rounded
                self.t_rearSpar_web_table[k] = t_rear_web_rounded
                self.t_Spar_cap_table[k] = t_cap_rounded
                self.sections[k].t_frontSpar_web = t_front_web_rounded
                self.sections[k].t_rearSpar_web = t_rear_web_rounded
                self.sections[k].t_Spar_cap = t_cap_rounded
                self.sections[k].spar_cap_width = spar_width

            else:
                raise ValueError('Wrong flag was given for optimizeSparThickness')
    def optimizeDcell(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm, dcell_c_buckling, dcell_s_buckling):
        '''
        Sets up tables of dcell thicknesses and modifies section attributes for all sections
        Inputs:
        Maximum load factor considered
        Single wing lift for the maximum load factor considered [N]
        Minimum load factor considered
        Single wing lift for the minimum load factor considered [N]
        Single engine mass [kg]
        Single engine thrust [N]
        Dynamic pressure q [kg.m^-1.s^-2]
        Aero moment coefficient Cm
        Object of the Dataseries class containing data for **** buckling of curved panels (ESDU *****)       
        '''
        SF_maxload,BM_maxload = self.SF_BM(max_load_factor, max_lift, engine_mass)
        SF_minload,BM_minload = self.SF_BM(min_load_factor, min_lift, engine_mass)
        T_maxload = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        T_minload = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        t_lower_current = 0
        t_mid_current = 0
        t_upper_current = 0
        dt = 1e-4  # All thicknesses will be rounded up to the next multiple of 0.1mm
        start_index = 0
        for k in range(self.Nsections):
            # Geometry
            section = self.sections[k]
            chord = section.chord
            height = 0.1013*chord
            wingbox_width = (self.ratio_rearSpar - self.ratio_frontSpar) * chord
            wingbox_area = height*wingbox_width
            stringer_spacing_upper = wingbox_width/(section.n_stringers_upper+1)# Spacing between two stringers
            stringer_spacing_lower = wingbox_width/(section.n_stringers_lower+1)# Spacing between two stringers
            t_eff_upper = section.t_skin_upper + section.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
            t_eff_lower = section.t_skin_lower + section.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
            I_box = (4*section.spar_cap_width*section.t_Spar_cap + (t_eff_upper + t_eff_lower)*wingbox_width) * (height/2)**2
            dcell_length = section.ratio_frontSpar*chord
            dcell_area = (np.pi*dcell_length*height)/2
            rib_spacing = section.rib_spacing
            r_lower = height*3 # Approximation
            r_mid = height/2 # Approximation
            r_upper = height*3 # Approximation
            lower_length = dcell_length # Approximation
            mid_length = height # Approximation
            upper_length = dcell_length # Approximation
            # Skin Material
            E = self.skin_material.E
            sigma_y = self.skin_material.sigma_y
            # Shear buckling coefficients
            ks_ratio_lower = rib_spacing / lower_length
            x_ks_lower = min(rib_spacing, lower_length)/np.sqrt(r_lower*section.t_dcell_lower)
            ks_lower = dcell_s_buckling.find_y(x_ks_lower, ks_ratio_lower)
            ks_ratio_mid = rib_spacing / mid_length
            x_ks_mid = min(rib_spacing, mid_length)/np.sqrt(r_mid*section.t_dcell_mid)
            ks_mid = dcell_s_buckling.find_y(x_ks_mid, ks_ratio_mid)
            ks_ratio_upper = rib_spacing / upper_length
            x_ks_upper = min(rib_spacing, upper_length)/np.sqrt(r_upper*section.t_dcell_upper)
            ks_upper = dcell_s_buckling.find_y(x_ks_upper, ks_ratio_upper)
            # Compression buckling coefficients
            x_kc_lower = rib_spacing**2 / (r_lower*section.t_dcell_lower)
            kc_lower = dcell_c_buckling.find_y(x_kc_lower)
            x_kc_mid = rib_spacing**2 / (r_mid*section.t_dcell_mid)
            kc_mid = dcell_c_buckling.find_y(x_kc_mid)
            x_kc_upper = rib_spacing**2 / (r_upper*section.t_dcell_upper)
            kc_upper = dcell_c_buckling.find_y(x_kc_upper)
            # Loads
            BM_maxload_sec = BM_maxload[k]
            BM_minload_sec = BM_minload[k]
            T_maxload_sec = T_maxload[k]
            T_minload_sec = T_minload[k]
            # SHARED VALUES BETWEEN ALL PANELS
            t_crit_s_yield_min = np.sqrt(3)*np.abs(T_minload_sec)/(2*(wingbox_area+dcell_area)*sigma_y)
            t_crit_s_yield_max = np.sqrt(3)*np.abs(T_maxload_sec)/(2*(wingbox_area+dcell_area)*sigma_y)
            # LOWER PANEL
            # Lower panel sizing for n=nmin
            lower_t_crit_s_buckling_min = ( np.abs(T_minload_sec)*min(rib_spacing, lower_length)**2 / (2*ks_lower*E*(wingbox_area+dcell_area)) )**(1/3)
            lower_t_crit_c_buckling = min(rib_spacing,lower_length)*np.sqrt(np.abs(BM_minload_sec)*height/(2*I_box*kc_lower*E)) # Only min load factor because lower skin is only in compression then
            lower_t_crit_axial_min = np.abs(BM_minload_sec)*height/(np.pi*r_lower**3 * sigma_y) - 2*I_box/(np.pi*r_lower**3)
            # Lower panel sizing for n=nmax
            lower_t_crit_s_buckling_max = ( np.abs(T_maxload_sec)*min(rib_spacing, lower_length)**2 / (2*ks_lower*E*(wingbox_area+dcell_area)) )**(1/3)
            lower_t_crit_axial_max = np.abs(BM_maxload_sec)*height/(np.pi*r_lower**3 * sigma_y) - 2*I_box/(np.pi*r_lower**3)
            # Lower panel final sizing
            t_lower = max(lower_t_crit_s_buckling_min, t_crit_s_yield_min, lower_t_crit_c_buckling, lower_t_crit_s_buckling_max, t_crit_s_yield_max, lower_t_crit_axial_min, lower_t_crit_axial_max, 0.003)
            # MID PANEL
            # Mid panel sizing for n=nmin
            mid_t_crit_s_buckling_min = ( np.abs(T_minload_sec)*min(rib_spacing, mid_length)**2 / (2*ks_mid*E*(wingbox_area+dcell_area)) )**(1/3)
            mid_t_crit_c_buckling_min = min(rib_spacing,mid_length)*np.sqrt(np.abs(BM_minload_sec)*height/(4*I_box*kc_mid*E))
            mid_t_crit_axial_min = np.abs(BM_minload_sec)*height/(np.pi*r_mid**3 * sigma_y) - 2*I_box/(np.pi*r_mid**3)
            # Mid panel sizing for n=nmax
            mid_t_crit_s_buckling_max = ( np.abs(T_maxload_sec)*min(rib_spacing, mid_length)**2 / (2*ks_mid*E*(wingbox_area+dcell_area)) )**(1/3)
            mid_t_crit_c_buckling_max = min(rib_spacing,mid_length)*np.sqrt(np.abs(BM_maxload_sec)*height/(4*I_box*kc_mid*E))
            mid_t_crit_axial_max = np.abs(BM_maxload_sec)*height/(np.pi*r_mid**3 * sigma_y) - 2*I_box/(np.pi*r_mid**3)
            # Mid panel final sizing
            t_mid = max(t_crit_s_yield_min, t_crit_s_yield_max, mid_t_crit_s_buckling_min, mid_t_crit_c_buckling_min, mid_t_crit_s_buckling_max, mid_t_crit_c_buckling_max,mid_t_crit_axial_min, mid_t_crit_axial_max, 0.003)
            # UPPER PANEL
            # Upper panel sizing for n=nmin
            upper_t_crit_s_buckling_min = ( np.abs(T_minload_sec)*min(rib_spacing, upper_length)**2 / (2*ks_upper*E*(wingbox_area+dcell_area)) )**(1/3)
            upper_t_crit_axial_min = np.abs(BM_minload_sec)*height/(np.pi*r_upper**3 * sigma_y) - 2*I_box/(np.pi*r_upper**3)
            # Upper panel sizing for n=nmax
            upper_t_crit_s_buckling_max = ( np.abs(T_maxload_sec)*min(rib_spacing, upper_length)**2 / (2*ks_upper*E*(wingbox_area+dcell_area)) )**(1/3)
            upper_t_crit_c_buckling = min(rib_spacing,upper_length)*np.sqrt(np.abs(BM_maxload_sec)*height/(2*I_box*kc_upper*E)) # Only max load factor because upper skin is only in compression then
            upper_t_crit_axial_max = np.abs(BM_maxload_sec)*height/(np.pi*r_upper**3 * sigma_y) - 2*I_box/(np.pi*r_upper**3)
            # Upper panel final sizing
            t_upper = max(t_crit_s_yield_min, t_crit_s_yield_max, upper_t_crit_s_buckling_min, upper_t_crit_s_buckling_max, upper_t_crit_c_buckling, upper_t_crit_axial_min, upper_t_crit_axial_max, 0.003)
            # UPDATING THICKNESSES
            # Lower panel
            if t_lower > t_lower_current:
                t_lower_current = t_lower
            # Mid panel
            if t_mid > t_mid_current:
                t_mid_current = t_mid
            # Upper panel
            if t_upper > t_upper_current:
                t_upper_current = t_upper
            # Checking if end of spar and assigning if necessary
            if section.rib_thickness > 0 or k == self.Nsections - 1:
                end_index = k
                t_lower_rounded = max(np.ceil(t_lower_current / dt) * dt, 0.003)
                t_mid_rounded  = max(np.ceil(t_mid_current / dt) * dt, 0.003)
                t_upper_rounded = max(np.ceil(t_upper_current / dt) * dt, 0.003)
                for i in range(start_index, end_index + 1):
                    self.t_dcell_lower_table[i] = t_lower_rounded
                    self.t_dcell_mid_table[i]  = t_mid_rounded
                    self.t_dcell_upper_table[i] = t_upper_rounded
                    self.sections[i].t_dcell_lower = t_lower_rounded
                    self.sections[i].t_dcell_mid = t_mid_rounded
                    self.sections[i].t_dcell_upper = t_upper_rounded
                t_lower_current = 0
                t_mid_current = 0
                t_upper_current = 0
                start_index = k+1


    def optimize_rib_thicknesses(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm, panel_s_buckling):
        '''
        Optimizing rib thicknesses for shear yield, shear buckling, and if necessary loads form engine pylon (heavy ribs)
        Inputs:
        Maximum load factor considered
        Single wing lift for the maximum load factor considered [N]
        Minimum load factor considered
        Single wing lift for the minimum load factor considered [N]
        Single engine mass [kg]      
        '''
        # Computing loads along the wing
        torque_max = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        torque_min = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        # Loop on ribs
        rib_index = 0
        for i in range(self.Nsections):
            section = self.sections[i]
            if section.rib_thickness > 0: # Section with ribs have a default non-zero rib thickness
                section_index = i
                # Computing geometry
                wingbox_height = 0.1013*section.chord
                wingbox_width = section.chord * (self.ratio_rearSpar-self.ratio_frontSpar)
                wingbox_area = wingbox_height*wingbox_width
                dcell_area = (np.pi*section.chord*section.ratio_frontSpar*wingbox_height)/2
                x_ks_mainrib = wingbox_height / wingbox_width # ESDU 71005
                ks_mainrib = panel_s_buckling.find_y(x_ks_mainrib)
                x_ks_dcellrib = wingbox_height / (section.chord*section.ratio_frontSpar)
                ks_dcellrib = panel_s_buckling.find_y(x_ks_dcellrib)
                # Material properties
                sigma_y = self.rib_material.sigma_y
                E = self.rib_material.E
                # MAIN RIB
                # Computing critical thickness for shear yield using Von Mises criteria
                torque_rib = max(np.abs(torque_max[section_index]), np.abs(torque_min[section_index]))                
                t1 = np.sqrt(3)*torque_rib/(2*(wingbox_area+dcell_area) * sigma_y)
                # Computing critical thickness for shear buckling
                t2_mainrib = ((wingbox_height**2 * torque_rib)/(2*ks_mainrib*E*(wingbox_area+dcell_area)))**(1/3)
                # DCELL RIB
                # Shear Bucklin (shear yield is same as main rib)
                t2_dcellrib = ((wingbox_height**2 * torque_rib)/(2*ks_dcellrib*E*(wingbox_area+dcell_area)))**(1/3)
                # Setting up min thickness
                if i==0 or section.engine:
                    t_min = 0.03
                else:
                    t_min = 0.002
                # ASSIGNING VALUES
                # Keeping the maximum thickness between the two thicknesses, and rounding up to 0.1mm
                t_mainrib = max(t1, t2_mainrib, t_min)
                t_dcellrib = max(t1, t2_dcellrib, t_min)
                dt = 0.0001 # Round up to 0.1mm
                t_mainrib_rounded = np.ceil(t_mainrib / dt) * dt + dt
                t_dcellrib_rounded = np.ceil(t_dcellrib / dt) * dt + dt
                # Assigning to the section
                section.rib_thickness = t_mainrib_rounded
                self.rib_t_table[rib_index] = t_mainrib_rounded
                section.dcell_rib_thickness = t_dcellrib_rounded
                # Increasing rib_index
                rib_index = rib_index + 1


    def check_wing(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm, panel_c_buckling, local_buckling, panel_s_buckling, dcell_c_buckling, dcell_s_buckling):
        '''
        Checks the structural integrity of the wing
        Inputs:
        Maximum load factor considered
        Single wing lift for the maximum load factor considered [N]
        Minimum load factor considered
        Single wing lift for the minimum load factor considered [N]
        Single engine mass [kg]
        Single engine thrust [N]
        Dynamic pressure q [kg.m^-1.s^-2]
        Aero moment coefficient Cm
        Object of the class dataseries containing the ESDU data for compressive buckling of flat panels (EDSU 72019)
        Object of the class dataseries containing the ESDU data for local buckling of Z stringers (EDSU 71014)
        Object of the class dataseries containing the ESDU data for shear buckling of flat panels (EDSU 71005)
        Object of the class dataseries containing the ESDU data for compression buckling of curved panels (EDSU 02.01.10) 
        Object of the class dataseries containing the ESDU data for shear buckling of curved panels (EDSU 02.03.18 and 02.03.19)    
        '''
        # Initialize variables
        self.failed_upper_sections = [0, []]
        self.failed_lower_sections = [0, []]
        self.failed_spars = [0, []]
        self.failed_dcells = [0, []]
        self.failed_ribs = [0, []]
        wing_lower_pass = True # Full wing flags
        wing_upper_pass = True # Full wing flags
        wing_spar_pass = True # Full wing flags
        wing_dcell_pass = True # Full wing flags
        wing_rib_pass = True # Full wing flags
        # Compute loads
        SF_maxload,BM_maxload = self.SF_BM(max_load_factor, max_lift, engine_mass)
        SF_minload,BM_minload = self.SF_BM(min_load_factor, min_lift, engine_mass)
        torque_maxload = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        torque_minload = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        # Loop on sections
        for k in range(self.Nsections-1): # Intentionally skipping tip section because tip load is zero
            BM_minload_sec = BM_minload[k]
            BM_maxload_sec = BM_maxload[k]
            torque_minload_sec = torque_minload[k]
            torque_maxload_sec = torque_maxload[k]
            SF_maxload_sec = SF_maxload[k]
            SF_minload_sec = SF_minload[k]
            section = self.sections[k]
            # Check skin and stringers
            lower_pass_min,upper_pass_min = section.skin_stress_checks(BM_minload_sec, torque_minload[k], panel_c_buckling, local_buckling, panel_s_buckling)
            lower_pass_max,upper_pass_max = section.skin_stress_checks(BM_maxload_sec, torque_maxload[k], panel_c_buckling, local_buckling, panel_s_buckling)
            section_upper_pass = upper_pass_min and upper_pass_max
            section_lower_pass = lower_pass_min and lower_pass_max
            if not section_lower_pass:
                wing_lower_pass = False
                self.failed_lower_sections[0] = self.failed_lower_sections[0] + 1
                self.failed_lower_sections[1].append(section.y)
            if not section_upper_pass:
                wing_upper_pass = False
                self.failed_upper_sections[0] = self.failed_upper_sections[0] + 1
                self.failed_upper_sections[1].append(section.y)
            # Checks spars
            section_spar_pass_maxload = section.spars_checks(BM_maxload[k], SF_maxload[k], torque_maxload[k], panel_s_buckling)
            section_spar_pass_minload = section.spars_checks(BM_minload[k], SF_minload[k], torque_minload[k], panel_s_buckling)
            if (not section_spar_pass_maxload) or (not section_spar_pass_minload):
                wing_spar_pass = False
                self.failed_spars[0] = self.failed_spars[0] + 1
                self.failed_spars[1].append(section.y)
            # Check DCell
            section_dcell_pass_maxload = section.dcell_checks(BM_maxload_sec, torque_maxload_sec, dcell_c_buckling, dcell_s_buckling)
            section_dcell_pass_minload = section.dcell_checks(BM_minload_sec, torque_minload_sec, dcell_c_buckling, dcell_s_buckling)
            if (not section_dcell_pass_maxload) or (not section_dcell_pass_minload):
                wing_dcell_pass = False
                self.failed_dcells[0] = self.failed_dcells[0] + 1
                self.failed_dcells[1].append(section.y)
            # Check Ribs
            section_rib_pass_maxload = section.rib_check(torque_maxload_sec, self.rib_material, panel_s_buckling)
            section_rib_pass_minload = section.rib_check(torque_minload_sec, self.rib_material, panel_s_buckling)
            if (not section_rib_pass_maxload) or (not section_rib_pass_minload):
                wing_rib_pass = False
                self.failed_ribs[0] = self.failed_ribs[0] + 1
                self.failed_ribs[1].append(section.y)
        return wing_lower_pass, wing_upper_pass, wing_spar_pass, wing_dcell_pass, wing_rib_pass

    def display_characteristics(self):
        '''
        Printing characteristics of the wing
        '''
        # Printing everything relevant
        print("Skin material: ", self.skin_material.name)
        print("Spar material: ", self.spar_material.name)
        print("Stringer material: ", self.stringer_material.name)
        print("Stringer area: ", self.stringer_area)
        print("Stringer pitch lower: ", self.stringer_pitch_lower)
        print("Stringer pitch upper: ", self.stringer_pitch_upper)
        print("Rib material: ",self.rib_material.name )
        print("Rib stations: ",self.rib_stations)
        print("DCell rib thickness", self.DCell_rib_t)

        # Plotting relevant values along span
        x_data = [self.sections[k].y for k in range(self.Nsections)]
        skin_t_upper = [self.sections[k].t_skin_upper for k in range(self.Nsections)]
        skin_t_lower = [self.sections[k].t_skin_lower for k in range(self.Nsections)]
        fig, axs = plt.subplots(5, 2, sharex=True, figsize=(9, 11))
        axs = axs.flatten()
        axs[0].scatter(x_data, skin_t_upper); axs[0].set_ylabel("Upper skin thickness"); axs[0].grid(True)
        axs[1].scatter(x_data, skin_t_lower); axs[1].set_ylabel("Lower skin thickness"); axs[1].grid(True)
        axs[2].scatter(x_data, self.t_dcell_lower_table); axs[2].set_ylabel("Dcell lower thickness"); axs[2].grid(True)
        axs[3].scatter(x_data, self.t_dcell_mid_table); axs[3].set_ylabel("Dcell mid thickness"); axs[3].grid(True)
        axs[4].scatter(x_data, self.t_dcell_upper_table); axs[4].set_ylabel("Dcell upper thickness"); axs[4].grid(True)
        axs[5].scatter(x_data, self.t_frontSpar_web_table); axs[5].set_ylabel("Front Spar thickness"); axs[5].set_xlabel("Y station"); axs[5].grid(True)
        axs[6].scatter(x_data, self.t_rearSpar_web_table); axs[6].set_ylabel("Rear Spar thickness"); axs[6].grid(True)
        axs[7].scatter(x_data, self.t_Spar_cap_table); axs[7].set_ylabel("Spars cap thickness"); axs[7].grid(True)
        axs[8].scatter(x_data, self.stringer_t_table); axs[8].set_ylabel("Stringer thickness"); axs[8].grid(True)
        axs[9].scatter(self.rib_stations, self.rib_t_table); axs[9].set_ylabel("Rib thickness"); axs[9].set_xlabel("Y station"); axs[9].grid(True)

        plt.tight_layout()
        plt.show()
    
    def plot_farar(self, farar_table):
        '''
        Plotting farar coefficients along the span of the wing for upper and lower sections
        '''
        farar_lower_table = []
        farar_upper_table = []
        for k in range(self.Nsections):
            farar_lower = self.sections[k].assign_farar_lower(farar_table)
            farar_upper = self.sections[k].assign_farar_upper(farar_table)
            farar_lower_table.append(farar_lower)
            farar_upper_table.append(farar_upper)
        x = [k for k in range(self.Nsections)]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

        # Lower
        axs[0].plot(x, farar_lower_table)
        axs[0].set_ylabel("Farar coefficients for lower skin")
        axs[0].grid(True)

        # Upper
        axs[1].plot(x, farar_upper_table)
        axs[1].set_ylabel("Farar coefficients for upper skin")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()
    
    def mass_split(self):
        skin_mass_upper = 0
        skin_mass_lower = 0
        stringer_mass_upper = 0
        stringer_mass_lower = 0
        spar_web_mass = 0
        spar_cap_mass = 0
        rib_mass = 0
        dcell_skin_mass = 0
        dy = self.sections[1].y - self.sections[0].y
        for section in self.sections:
            height = 0.1013*section.chord
            width = section.chord * (section.ratio_rearSpar - section.ratio_frontSpar)
            skin_mass_upper += section.t_skin_upper*width*dy*self.skin_material.rho
            skin_mass_lower += section.t_skin_lower*width*dy*self.skin_material.rho
            stringer_mass_upper += section.n_stringers_upper * section.stringer_area * dy * self.stringer_material.rho
            stringer_mass_lower += section.n_stringers_lower * section.stringer_area * dy * self.stringer_material.rho
            spar_web_mass += (section.t_frontSpar_web + section.t_rearSpar_web) * height * dy * self.skin_material.rho
            spar_cap_mass += 4*section.spar_cap_width*section.t_Spar_cap*dy * self.spar_material.rho
            rib_mass += section.rib_thickness * height*width + section.dcell_rib_thickness*(np.pi/2 * section.chord*section.ratio_frontSpar * height) * self.rib_material.rho
            dcell_skin_mass += ((section.t_dcell_lower + section.t_dcell_upper) *section.ratio_frontSpar*section.chord + section.t_dcell_mid*.1013*section.chord)*dy * self.skin_material.rho
        print("Upper skin mass: ", skin_mass_upper)
        print("Lower skin mass: ", skin_mass_lower)
        print("Upper stringer mass: ", stringer_mass_upper)
        print("Lower stringer mass: ", stringer_mass_lower)
        print("Spar web mass: ", spar_web_mass)
        print("Spar cap mass: ", spar_cap_mass)
        print("Rib mass: ", rib_mass)
        print("Dcell skin mass: ", dcell_skin_mass)

    def pruning_check(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm, panel_c_buckling, local_buckling, panel_s_buckling):
        SF_maxload, BM_maxload = self.SF_BM(max_load_factor, max_lift, engine_mass)
        T_maxload = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        SF_minload, BM_minload = self.SF_BM(min_load_factor, min_lift, engine_mass)
        T_minload = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        root_section = self.sections[0]
        pass_flag_1, pass_flag_2 = root_section.skin_stress_checks(BM_maxload[0], T_maxload[0], panel_c_buckling, local_buckling, panel_s_buckling, pruning=True)
        pass_flag_3, pass_flag_4 = root_section.skin_stress_checks(BM_minload[0], T_minload[0], panel_c_buckling, local_buckling, panel_s_buckling, pruning=True)

        return (pass_flag_1 and pass_flag_2 and pass_flag_3 and pass_flag_4)
    
    def max_skin_stress(self, max_load_factor, max_lift, min_load_factor, min_lift, engine_mass, engine_thrust, q, Cm):
        # Computing loads
        SF_max,BM_max = self.SF_BM(max_load_factor, max_lift, engine_mass)
        SF_min,BM_min = self.SF_BM(min_load_factor, min_lift, engine_mass)
        T_max = self.torque(max_load_factor, max_lift, q, Cm, engine_mass, engine_thrust)
        T_min = self.torque(min_load_factor, min_lift, q, Cm, engine_mass, engine_thrust)
        # Initialising
        max_skin_stress = 0
        # Looping over sections
        for k in range(self.Nsections):
            sec_skin_stress = max(self.sections[k].max_skin_stress_section(BM_max[k], T_max[k]), self.sections[k].max_skin_stress_section(BM_min[k], T_min[k]))
            if sec_skin_stress > max_skin_stress:
                max_skin_stress = sec_skin_stress
        return max_skin_stress
