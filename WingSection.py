
import numpy as np
from farar import fararserie

class wingSection:
    def __init__(self, y, chord, engine, skin_material, t_skin_lower, t_skin_upper, spar_material, ratio_frontSpar, t_frontSpar_web, ratio_rearSpar, t_rearSpar_web, t_Spar_cap, spar_cap_width, stringer_area, stringer_t, stringer_material, n_stringers_upper, n_stringers_lower, rib_spacing, rib_thickness, t_dcell_lower, t_dcell_mid, t_dcell_upper, dcell_rib_thickness):
        self.y = y # Y station of the section, from the plane of symmetry [m]
        self.chord = chord # Chord of the section [m]
        self.engine = engine # Boolean, is True if there is an engine on that specific section, False otherwise
        self.skin_material = skin_material # Material of the skin of the section, object of the class material
        self.t_skin_lower = t_skin_lower # Thickness of the lower skin [m]
        self.t_skin_upper = t_skin_upper # Thickness of the upper skin [m]
        self.spar_material = spar_material # Spar material, object of the class material
        self.ratio_frontSpar = ratio_frontSpar # X position of front spar as a ratio of chord
        self.t_frontSpar_web = t_frontSpar_web # Thickness of the front spar [m]
        self.ratio_rearSpar = ratio_rearSpar # X position of rear spar as a ratio of chord
        self.t_rearSpar_web = t_rearSpar_web # Thickness of the rear spar [m]
        self.t_Spar_cap = t_Spar_cap # Spar cap thickness [m]
        self.spar_cap_width = spar_cap_width # Spar cap width [m]
        self.stringer_area = stringer_area # Aera of stringer [m²]
        self.stringer_t = stringer_t # Stringer thickness [m]
        self.stringer_material = stringer_material # Stringer material
        self.n_stringers_upper = n_stringers_upper # Number of stringers on the upper part of the section
        self.n_stringers_lower = n_stringers_lower # Number of stringers on the lower part of the section
        self.rib_spacing = rib_spacing # Rib spacing for this specific section [m]
        self.rib_thickness = rib_thickness # Rib thickness if there is a rib [m]
        self.t_dcell_lower = t_dcell_lower # DCell lower skin thickness [m]
        self.t_dcell_mid = t_dcell_mid # DCell mid skin thickness [m]
        self.t_dcell_upper = t_dcell_upper # DCell upper skin thickness [m]
        self.dcell_rib_thickness = dcell_rib_thickness # DCell rib thickness [m]
        self.structure_pass_flag = False # Flag showing whether the section passes or fails structural checks
    

    def mass_per_span(self, span, Nsections):
        '''
        Computing section mass per span for distributed masses only
        Inputs:
        Span of the wing, from root to tip [m]
        Number of sections used to discretise the wing
        '''
        # Computing masses
        mass_stringers = self.stringer_material.rho * self.stringer_area * (self.n_stringers_lower + self.n_stringers_upper)
        mass_skin_wingbox = self.skin_material.rho * ((self.t_skin_lower + self.t_skin_upper) * (self.ratio_rearSpar-self.ratio_frontSpar)*self.chord)
        mass_spars_skin = self.skin_material.rho * 0.1013 * (self.t_frontSpar_web + self.t_rearSpar_web) * self.chord # The 0.1013 is from profile coordinates.
        mass_spars_cap = 4 * self.spar_cap_width * self.t_Spar_cap * self.spar_material.rho
        mass_skin_dcell = self.skin_material.rho * ((self.t_dcell_lower + self.t_dcell_upper) *self.ratio_frontSpar*self.chord + self.t_dcell_mid*.1013*self.chord) # Dcell length approximated as a rectangular section
        # Mass of components that are not part of the wing box - Assumed to be 20% of the wing mass provided by the brief, uniformly distributed
        mass_others = 0.2*4632/2 / span
        # Final mass per span
        mass_per_span = mass_stringers + mass_skin_wingbox + mass_spars_skin + mass_spars_cap + mass_skin_dcell + mass_others
        return mass_per_span

    def CoG(self, span):
        '''
        Computing X station of the distributed masses for the section CoG, from LE station
        '''
        # Linear masses of all components
        mass_skin_wingbox = self.skin_material.rho * ((self.t_skin_lower + self.t_skin_upper) * (self.ratio_rearSpar-self.ratio_frontSpar)*self.chord)
        mass_stringers = self.stringer_material.rho * self.stringer_area * (self.n_stringers_lower + self.n_stringers_upper)
        mass_frontSpar = self.skin_material.rho*0.1013*self.t_frontSpar_web*self.chord + self.spar_material.rho*self.spar_cap_width*self.t_Spar_cap # 0.1013*c is the thickness at the front spar location for the wing profile given in the brief
        mass_rearSpar = self.skin_material.rho*0.1013*self.t_rearSpar_web*self.chord + self.spar_material.rho*self.spar_cap_width*self.t_Spar_cap # 0.1013*c is the thickness at the front spar location for the wing profile given in the brief
        mass_skin_dcell = self.skin_material.rho * ((self.t_dcell_lower + self.t_dcell_upper) *self.ratio_frontSpar*self.chord + self.t_dcell_mid*.1013*self.chord) # Dcell length approximated as a rectangular section
        mass_others = 0.2*4632/2 / span
        # Computing the x position fo the CoG
        x_CoG = ((self.ratio_rearSpar - self.ratio_frontSpar) * self.chord * (mass_skin_wingbox + mass_stringers)
                 + self.ratio_frontSpar * mass_frontSpar * self.chord
                 + self.ratio_rearSpar * mass_rearSpar * self.chord
                 + self.ratio_frontSpar*2/3 * mass_skin_dcell*self.chord # DCell CoG assumed at 2/3 of the distance from LE to front spar
                 + self.chord * (self.ratio_rearSpar + 1/3 * (1-self.ratio_rearSpar)) * mass_others # Other masses assumed at 1/3 of the distance from rear spar to TE
                 )/(mass_skin_wingbox + mass_stringers + mass_frontSpar + mass_rearSpar + mass_skin_dcell + mass_others)
        return x_CoG

    def load(self, load_factor, span, y_root, Nsections, lift, engine_mass, rib_material):
        '''
        Computing load on the wing section, in N
        Inputs:
        Load factor considered
        Single wing span [m]
        Number of wing sections
        Single wing lift [N]
        '''
        # Boundaries of wing section
        y_min = self.y - span/(2*(Nsections-1))
        y_max = self.y + span/(2*(Nsections-1))
        # Distributed masses
        V_ymin = - self.mass_per_span(span, Nsections) * 9.81 * load_factor + 4*lift*np.sqrt(1-((y_min-y_root)/span)**2)/(np.pi*span)
        V_ymax = - self.mass_per_span(span, Nsections) * 9.81 * load_factor + 4*lift*np.sqrt(1-(min(y_max-y_root,span)/span)**2)/(np.pi*span)
        # Engine mass
        mass_engine = 0
        if self.engine:
            mass_engine = engine_mass
        inertia_engine = -9.81 * mass_engine * load_factor
        # Rib mass - Rib thickness is zero if there is no rib at this station
        mass_rib = self.rib_thickness * self.chord * (self.ratio_rearSpar - self.ratio_frontSpar) * 0.1013*self.chord * rib_material.rho
        inertia_rib = -9.81 * mass_rib * load_factor
        # Dcell rib mass - Rib thickness is zero if there is no rib at this station
        dcell_rib_area = np.pi/2 * self.chord*0.1013 * self.chord*self.ratio_frontSpar # DCell area approximated as a half ellipse
        mass_dcell_rib = self.dcell_rib_thickness * dcell_rib_area * rib_material.rho
        inertia_dcell_rib = -9.81 * mass_dcell_rib * load_factor
        # Final load computation
        Vy = (y_max - y_min)*(V_ymin + V_ymax)/2 + inertia_engine + inertia_rib + inertia_dcell_rib
        return Vy

    def torque_section(self, load_factor, span, y_root, Nsections, q, Cm, lift, engine_mass, engine_thrust):
        '''
        Computing torque on the wing section
        Inputs:
        Load factor considered
        Single wing span [m]
        Number of sections
        Dynamic pressure q [kg.m^-1.s^-2]
        Aerodynamic moment coefficient Cm
        Single wing lift [N]
        '''
        # Flexural axis position
        x_flex = self.chord * (self.ratio_rearSpar + self.ratio_frontSpar)/2
        # Section boundaries
        y_min = self.y - span/(2*(Nsections-1))
        y_max = self.y + span/(2*(Nsections-1))
        # Lift load at this section, assuming an elliptical distribution
        lift_load = (y_max - y_min) * (4*lift*np.sqrt(1-((y_min-y_root)/span)**2)/(np.pi*span) + 4*lift*np.sqrt(1-(min(y_max-y_root,span)/span)**2)/(np.pi*span))/2
        # Inertia load
        inertia_load = -9.81 * load_factor * self.mass_per_span(span, Nsections) * (y_max - y_min)
        # Moments around the flexural axis
        m_lift = lift_load * (x_flex - self.chord/4)
        m_inertia = inertia_load * (x_flex - self.CoG(span))
        m_aero = 1/2 * q * self.chord**2 * Cm * (y_max - y_min)
        m_engine = 0
        m_thrust = 0
        if self.engine:
            m_engine = - x_flex * 9.81 * load_factor * engine_mass # Assumes the engine weight is applied at the LE station (see brief's pictures)
            m_thrust = 0.48 * engine_thrust #0.48m is the radius of the nacelle, see brief
        t = m_lift + m_inertia + m_aero + m_engine + m_thrust
        return t

    def assign_farar_upper(self, farar_table):
        '''
        Assigning the value of the farar coefficient for the upper skin ofthis section based on data extracted from the graph in excel (ESDU)
        Inputs: table of objects of the fararserie class
        '''
        # Computing wing box width
        width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar)
        # Coordinates on the farar graph
        x = self.stringer_area / self.t_skin_upper / (width / (self.n_stringers_upper + 1))
        y = self.stringer_t / self.t_skin_upper
        # Initialising values
        min = 1e9
        farar_value = 10
        # Looking for closest curve on the graph
        for fararserie in farar_table:
            dist = fararserie.min_dist([x,y])[0]
            if dist < min:
                min = dist
                farar_value = fararserie.value
        return farar_value
    
    def assign_farar_lower(self, farar_table):
        '''
        Assigning the value of the farar coefficient for the lower skin ofthis section based on data extracted from the graph in excel (ESDU)
        Inputs: table of objects of the fararserie class
        '''
        width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar) #width of wing box
        x = self.stringer_area / self.t_skin_lower / (width / (self.n_stringers_lower + 1))
        y = self.stringer_t / self.t_skin_lower
        min = 1e9
        farar_value = 10
        for fararserie in farar_table:
            dist = fararserie.min_dist([x,y])[0]
            if dist < min:
                min = dist
                farar_value = fararserie.value
        return farar_value
    


    def Ks_coeff(self, Ks_data):
        '''
        Returning the value of the Ks coefficient for spar design
        Inputs: Table of points extracted from ESDU graph
        '''
        Ks = None
        a = self.rib_spacing
        b = self.chord * 0.1013
        # X coordinate on graph
        x_axis = a/b
        if x_axis < 1:
            x_axis = b/a
        # If out of graph, capping at end value
        if x_axis >= Ks_data[len(Ks_data)-1][0]:
            Ks = 8.1
        else:
            k = 0
            while Ks_data[k][0] < x_axis:
                k = k + 1
            if np.abs(Ks_data[k-1][0] - x_axis) < np.abs(Ks_data[k][0] - x_axis):
                Ks = Ks_data[k-1][1]
            else:
                Ks = Ks_data[k][1]
        return Ks

    def skin_stress_checks_upper(self, Mx, T, panel_c_buckling, local_buckling, panel_s_buckling, pruning=False):
        '''
        Computing bending stress from full wing bending moment distribution
        Inputs:
        Bending moment Mx at this specific section [Nm]
        Torque at this section [Nm] ; space between ribs [m]
        Object of the class dataseries containing the ESDU data for compressive buckling of flat panels (EDSU 72019)
        Object of the class dataseries containing the ESDU data for local buckling of Z stringers (EDSU 71014)
        Object of the class dataseries containing the ESDU data for shear buckling of flat panels (EDSU 71005)
        '''
        # Geometrical properties and coefficients
        width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar) #width of wing box
        height = self.chord * 0.1013 # Height of the box - Thickness at 15% of the chord
        stringer_spacing_upper = width/(self.n_stringers_upper+1)# Spacing between two stringers
        stringer_spacing_lower = width/(self.n_stringers_lower+1)# Spacing between two stringers
        t_eff_upper = self.t_skin_upper + self.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
        t_eff_lower = self.t_skin_lower + self.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
        # Axial stress due to bending
        I = (width*(t_eff_lower+t_eff_upper)*(height/2)**2
        + (self.t_frontSpar_web+self.t_rearSpar_web)*height**3/12
        + 4*self.spar_cap_width*self.t_Spar_cap*(height/2)**2)
        axial_stress_upper = np.abs(Mx)*(height/2)/I # Axial stress per unit length [N/m²] - See Textbook p516
        # Shear stress due to torque - Bredt-Batho
        shear_stress_upper = np.abs(T) / (2*width*height*t_eff_upper)
        # Initialisation of flag
        upper_pass = False
        # If pruning (ie preliminary checks to eliminate design that are very weak)
        if pruning:
            checking_coeff = 1.2
        else:
            checking_coeff = 1
        # Upper compression case
        if Mx > 0:
            # Compression Yield
            crit_c_yield_stress = self.skin_material.sigma_y
            # Global Buckling - ESDU72019
            x_panel_c_buckling = self.rib_spacing / stringer_spacing_upper            
            k_panel_c_buckling = panel_c_buckling.find_y(x_panel_c_buckling)
            f_0 = np.pi**2 * self.skin_material.E * (self.t_skin_upper/stringer_spacing_upper)**2 / (3*(1-self.skin_material.poisson**2))
            crit_buckling_global = k_panel_c_buckling * f_0
            # Local Buckling - ESDU71014
            ratio = self.stringer_t/self.t_skin_upper
            h = self.stringer_area / (1.6 * self.stringer_t) # Assumes d/h=0.3 (see ESDU nottaions)
            x_local_buckling = h/stringer_spacing_upper
            k_local_buckling = local_buckling.find_y(x_local_buckling, ratio)
            crit_buckling_local = k_local_buckling * self.stringer_material.E * (self.t_skin_upper/stringer_spacing_upper)**2
            # Shear yield - Von Mises criterion
            crit_s_yield_stress = self.skin_material.sigma_y/np.sqrt(3)
            # Shear buckling - ESDU71005
            x_shear_buckling = stringer_spacing_upper/self.rib_spacing
            if x_shear_buckling > 1:
                x_shear_buckling = 1/x_shear_buckling
            k_shear_buckling = panel_s_buckling.find_y(x_shear_buckling)
            crit_buckling_shear = k_shear_buckling * self.skin_material.E * (self.t_skin_upper/stringer_spacing_upper)**2
            # Critical stresses
            crit_axial = min(crit_c_yield_stress, crit_buckling_global, crit_buckling_local)
            crit_shear = min(crit_s_yield_stress, crit_buckling_shear)
            # Stress ratios
            R_b = axial_stress_upper / crit_axial
            R_s = shear_stress_upper / crit_shear
            # Checks
            if R_s**2 + R_b**2 < checking_coeff: # Other checks (R_b<1, R_s<1) are redundant
                upper_pass = True
        # Upper tension case
        else:
            # Tension yield
            critical_tension = self.skin_material.sigma_y
            # Shear yield - Von Mises criterion
            crit_s_yield_stress = self.skin_material.sigma_y/np.sqrt(3)
            # Shear buckling - ESDU71005
            x_shear_buckling = stringer_spacing_upper/self.rib_spacing
            if x_shear_buckling > 1:
                x_shear_buckling = 1/x_shear_buckling
            k_shear_buckling = panel_s_buckling.find_y(x_shear_buckling)
            crit_buckling_shear = k_shear_buckling * self.skin_material.E * (self.t_skin_upper/stringer_spacing_upper)**2
            # Critical stresses
            crit_axial = critical_tension
            crit_shear = min(crit_s_yield_stress, crit_buckling_shear)
            # Stress ratios
            R_b = axial_stress_upper / crit_axial
            R_s = shear_stress_upper / crit_shear
            # Checks
            if R_s**2 + R_b**2 < checking_coeff: # Other checks (R_b<1, R_s<1) are redundant
                upper_pass = True
        return upper_pass

    def skin_stress_checks_lower(self, Mx, T, panel_c_buckling, local_buckling, panel_s_buckling, pruning=False):
        '''
        Computing bending stress from full wing bending moment distribution
        Inputs:
        Bending moment Mx at this specific section [Nm]
        Torque at this section [Nm] ; space between ribs [m]
        Object of the class dataseries containing the ESDU data for compressive buckling of flat panels (EDSU 72019)
        Object of the class dataseries containing the ESDU data for local buckling of Z stringers (EDSU 71014)
        Object of the class dataseries containing the ESDU data for shear buckling of flat panels (EDSU 71005)        
        '''
        # Geometrical properties and coefficients
        width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar) #width of wing box
        height = self.chord * 0.1013 # Height of the box - Thickness at 15% of the chord
        stringer_spacing_upper = width/(self.n_stringers_upper+1)# Spacing between two stringers
        stringer_spacing_lower = width/(self.n_stringers_lower+1)# Spacing between two stringers
        t_eff_upper = self.t_skin_upper + self.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
        t_eff_lower = self.t_skin_lower + self.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
        # Axial stress due to bending
        I = (width*(t_eff_lower+t_eff_upper)*(height/2)**2
        + (self.t_frontSpar_web+self.t_rearSpar_web)*height**3/12
        + 4*self.spar_cap_width*self.t_Spar_cap*(height/2)**2)
        axial_stress_lower = np.abs(Mx)*(height/2)/I # Axial stress per unit length [N/m²] - See Textbook p339/340
        # Shear stress due to torque - Bredt-Batho
        shear_stress_lower = np.abs(T) / (2*width*height*t_eff_lower)
        # Initialisation of flag
        lower_pass = False
        # If pruning (ie preliminary checks to eliminate design that are very weak)
        if pruning:
            checking_coeff = 1.2
        else:
            checking_coeff = 1
        # Lower compression case
        if Mx < 0:
            # Compression Yield
            crit_c_yield_stress = self.skin_material.sigma_y
            # Global Buckling - ESDU72019
            x_panel_c_buckling = self.rib_spacing / stringer_spacing_lower            
            k_panel_c_buckling = panel_c_buckling.find_y(x_panel_c_buckling)
            f_0 = np.pi**2 * self.skin_material.E * (self.t_skin_lower/stringer_spacing_lower)**2 / (3*(1-self.skin_material.poisson**2))
            crit_buckling_global = k_panel_c_buckling * f_0
            # Local Buckling - ESDU71014
            ratio = self.stringer_t/self.t_skin_lower
            h = self.stringer_area / (1.6 * self.stringer_t) # Assumes d/h=0.3 (see ESDU nottaions)
            x_local_buckling = h/stringer_spacing_lower
            k_local_buckling = local_buckling.find_y(x_local_buckling, ratio)
            crit_buckling_local = k_local_buckling * self.stringer_material.E * (self.t_skin_lower/stringer_spacing_lower)**2
            # Shear yield - Von Mises criterion
            crit_s_yield_stress = self.skin_material.sigma_y/np.sqrt(3)
            # Shear buckling - ESDU71005
            x_shear_buckling = stringer_spacing_lower/self.rib_spacing
            if x_shear_buckling > 1:
                x_shear_buckling = 1/x_shear_buckling
            k_shear_buckling = panel_s_buckling.find_y(x_shear_buckling)
            crit_buckling_shear = k_shear_buckling * self.skin_material.E * (self.t_skin_lower/stringer_spacing_lower)**2
            # Critical stresses
            crit_axial = min(crit_c_yield_stress, crit_buckling_global, crit_buckling_local)
            crit_shear = min(crit_s_yield_stress, crit_buckling_shear)
            # Stress ratios
            R_b = axial_stress_lower / crit_axial
            R_s = shear_stress_lower / crit_shear
            # Checks (other checks are redundant)
            if R_s**2 + R_b**2 < checking_coeff:
                lower_pass = True
        # Lower tension case
        else:
            # Tension yield
            critical_tension = self.skin_material.sigma_y
            # Shear yield - Von Mises criterion
            crit_s_yield_stress = self.skin_material.sigma_y/np.sqrt(3)
            # Shear buckling - ESDU71005
            x_shear_buckling = stringer_spacing_lower/self.rib_spacing
            if x_shear_buckling > 1:
                x_shear_buckling = 1/x_shear_buckling
            k_shear_buckling = panel_s_buckling.find_y(x_shear_buckling)
            crit_buckling_shear = k_shear_buckling * self.skin_material.E * (self.t_skin_lower/stringer_spacing_lower)**2
            # Critical stresses
            crit_axial = critical_tension
            crit_shear = min(crit_s_yield_stress, crit_buckling_shear)
            # Stress ratios
            R_b = axial_stress_lower / crit_axial
            R_s = shear_stress_lower / crit_shear
            # Checks
            if R_s**2 + R_b**2 < checking_coeff: # Other checks are redundant
                lower_pass = True
        return lower_pass


    def skin_stress_checks(self, Mx, T, panel_c_buckling, local_buckling, panel_s_buckling, pruning=False):
        '''
        Checking for failure of skin and stringers for both the upper and lower side
        Inputs:
        Bending moment for this section [Nm]
        Torque for this section [Nm]
        Object of the class dataseries containing the ESDU data for compressive buckling of flat panels (EDSU 72019)
        Object of the class dataseries containing the ESDU data for local buckling of Z stringers (EDSU 71014)
        Object of the class dataseries containing the ESDU data for shear buckling of flat panels (EDSU 71005)   
        '''
        lower_pass = self.skin_stress_checks_lower(Mx, T, panel_c_buckling, local_buckling, panel_s_buckling, pruning)
        upper_pass = self.skin_stress_checks_upper(Mx, T, panel_c_buckling, local_buckling, panel_s_buckling, pruning)
        return (lower_pass, upper_pass)
    
    def spars_checks(self, M, Vy, torque, panel_s_buckling):
        '''
        Checks spars for compressive buckling, axial yield, shear buckling, shear yield
        Inputs:
        Bending Moment [Nm]
        Shear force [N]
        Torque [Nm]
        Object of the Dataseries class containing data for shear buckling (ESDU 71005)
        '''
        # Initialize
        front_spar_pass = False
        rear_spar_pass = False
        # Geometry/Material
        height = 0.1013*self.chord
        box_width = (self.ratio_rearSpar - self.ratio_frontSpar) *self.chord
        E_web = self.skin_material.E
        sigma_y_web = self.skin_material.sigma_y
        stringer_spacing_upper = box_width/(self.n_stringers_upper+1)# Spacing between two stringers
        stringer_spacing_lower = box_width/(self.n_stringers_lower+1)# Spacing between two stringers
        t_eff_upper = self.t_skin_upper + self.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
        t_eff_lower = self.t_skin_lower + self.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
        I_wingbox = (4*self.spar_cap_width*self.t_Spar_cap + (t_eff_upper + t_eff_lower)*box_width) * (height/2)**2
        E_cap = self.spar_material.E
        sigma_y_cap = self.spar_material.sigma_y
        # Applied stresses
        applied_shear_stress_front = np.abs(torque/box_width + Vy)/(2*self.t_frontSpar_web*height)
        applied_shear_stress_rear = np.abs(torque/box_width - Vy)/(2*self.t_rearSpar_web*height)
        applied_axial_stress = np.abs(M)*height/2 / I_wingbox # Same for front and rear
        # Spar cap buckling critical stress
        crit_buckling_c_stress = np.pi**2 * E_cap * self.t_Spar_cap**2 / (24*self.rib_spacing**2)
        # Spar cap yield critical stress
        crit_axial_yield = sigma_y_cap
        # Spar web shear buckling critical stresses
        x_ks = height / self.rib_spacing
        if x_ks > 1:
            x_ks = 1/x_ks
        Ks = panel_s_buckling.find_y(x_ks)
        crit_buckling_s_stress_front = Ks * E_web * (self.t_frontSpar_web/height)**2
        crit_buckling_s_stress_rear = Ks * E_web * (self.t_rearSpar_web/height)**2
        # Spar web shear yield critical stress - Von Mises criterion
        crit_shear_yield = sigma_y_web / np.sqrt(3)
        # Critical values
        crit_cap_stress = min(crit_buckling_c_stress, crit_axial_yield)
        crit_shear_stress_front_web = min(crit_shear_yield, crit_buckling_s_stress_front)
        crit_shear_stress_rear_web = min(crit_shear_yield, crit_buckling_s_stress_rear)
        # Checks
        if applied_axial_stress < crit_cap_stress and applied_shear_stress_front < crit_shear_stress_front_web :
            front_spar_pass = True
        if applied_axial_stress < crit_cap_stress and applied_shear_stress_rear < crit_shear_stress_rear_web :
            rear_spar_pass = True
        full_pass = front_spar_pass and rear_spar_pass
        '''
        if not full_pass:
            if applied_axial_stress < crit_cap_stress:
                print('Spar web failure')
            else:
                print('Spar cap failure')
        '''
        return(full_pass)
    

    def dcell_checks(self, Mx, torque, dcell_c_buckling, dcell_s_buckling):
        '''
        Checks for DCell failure
        Inputs:
        Bending Moment for this section [Nm]
        Torque for this section [Nm]
        Object of the class dataseries containing the ESDU data for compression buckling of curved panels (EDSU 02.01.10) 
        Object of the class dataseries containing the ESDU data for shear buckling of curved panels (EDSU 02.03.18 and 02.03.19) 
        '''
        # Initialising
        pass_flag = True
        # Material properties
        E = self.skin_material.E
        sigma_y = self.skin_material.sigma_y # Yield tensile strength
        # Geometrical properties
        chord = self.chord
        height = 0.1013*chord
        wingbox_width = (self.ratio_rearSpar - self.ratio_frontSpar) * chord
        wingbox_area = height*wingbox_width
        stringer_spacing_upper = wingbox_width/(self.n_stringers_upper+1)# Spacing between two stringers
        stringer_spacing_lower = wingbox_width/(self.n_stringers_lower+1)# Spacing between two stringers
        t_eff_upper = self.t_skin_upper + self.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
        t_eff_lower = self.t_skin_lower + self.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
        I_box = (4*self.spar_cap_width*self.t_Spar_cap + (t_eff_upper + t_eff_lower)*wingbox_width) * (height/2)**2
        dcell_length = self.ratio_frontSpar*chord
        dcell_area = (np.pi*dcell_length*height)/2
        rib_spacing = self.rib_spacing
        r_lower = height*3 # Approximation
        r_mid = height/2 # Approximation
        r_upper = height*3 # Approximation
        lower_length = dcell_length # Approximation
        mid_length = height # Approximation
        upper_length = dcell_length # Approximation
        I_box_and_dcell_approx_lower = I_box + np.pi*r_lower**3 * self.t_dcell_lower/2 # Approximation
        I_box_and_dcell_approx_mid = I_box + np.pi*r_mid**3 * self.t_dcell_mid/2 # Approximation
        I_box_and_dcell_approx_upper = I_box + np.pi*r_upper**3 * self.t_dcell_upper/2 # Approximation
        # Shear buckling coefficients
        ks_ratio_lower = rib_spacing / lower_length
        x_ks_lower = min(rib_spacing, lower_length)/np.sqrt(r_lower*self.t_dcell_lower)
        ks_lower = dcell_s_buckling.find_y(x_ks_lower, ks_ratio_lower)
        ks_ratio_mid = rib_spacing / mid_length
        x_ks_mid = min(rib_spacing, mid_length)/np.sqrt(r_mid*self.t_dcell_mid)
        ks_mid = dcell_s_buckling.find_y(x_ks_mid, ks_ratio_mid)
        ks_ratio_upper = rib_spacing / upper_length
        x_ks_upper = min(rib_spacing, upper_length)/np.sqrt(r_upper*self.t_dcell_upper)
        ks_upper = dcell_s_buckling.find_y(x_ks_upper, ks_ratio_upper)
        # Compression buckling coefficients
        x_kc_lower = rib_spacing**2 / (r_lower*self.t_dcell_lower)
        kc_lower = dcell_c_buckling.find_y(x_kc_lower)
        x_kc_mid = rib_spacing**2 / (r_mid*self.t_dcell_mid)
        kc_mid = dcell_c_buckling.find_y(x_kc_mid)
        x_kc_upper = rib_spacing**2 / (r_upper*self.t_dcell_upper)
        kc_upper = dcell_c_buckling.find_y(x_kc_upper)
        # Applied stress
        axial_stress_lower = np.abs(Mx)*height/2 / I_box_and_dcell_approx_lower
        axial_stress_mid = np.abs(Mx)*height/4 / I_box_and_dcell_approx_mid
        axial_stress_upper = np.abs(Mx)*height/2 / I_box_and_dcell_approx_upper
        shear_stress_lower = np.abs(torque)/(2*(wingbox_area+dcell_area)*self.t_dcell_lower)
        shear_stress_mid = np.abs(torque)/(2*(wingbox_area+dcell_area)*self.t_dcell_mid)
        shear_stress_upper = np.abs(torque)/(2*(wingbox_area+dcell_area)*self.t_dcell_upper)
        # Checks
        # If BM>0, upper compression and lower tension
        if Mx > 0:
            # All criteria
            crit_c_buckling_upper = kc_upper*E*(self.t_dcell_upper/min(rib_spacing,upper_length))**2
            crit_c_buckling_mid = kc_mid*E*(self.t_dcell_mid/min(rib_spacing,mid_length))**2
            crit_s_buckling_lower = ks_lower*E*(self.t_dcell_lower/min(rib_spacing,lower_length))**2
            crit_s_buckling_mid = ks_mid*E*(self.t_dcell_mid/min(rib_spacing,mid_length))**2
            crit_s_buckling_upper = ks_upper*E*(self.t_dcell_upper/min(rib_spacing,upper_length))**2
            crit_axial_yield = sigma_y
            crit_s_yield = sigma_y/np.sqrt(3)
            # Checks:
            # Lower
            if axial_stress_lower > crit_axial_yield or shear_stress_lower > min(crit_s_buckling_lower,crit_s_yield) :
                pass_flag = False
            # Mid
            if axial_stress_mid > min(crit_axial_yield,crit_c_buckling_mid) or shear_stress_mid > min(crit_s_buckling_mid,crit_s_yield):
                pass_flag = False
            # Upper
            if axial_stress_upper > min(crit_axial_yield,crit_c_buckling_upper) or shear_stress_upper > min(crit_s_buckling_upper,crit_s_yield):
                pass_flag = False
        else: # BM<0, lower in compression
            # All criteria
            crit_c_buckling_lower = kc_lower*E*(self.t_dcell_lower/min(rib_spacing,lower_length))**2
            crit_c_buckling_mid = kc_mid*E*(self.t_dcell_mid/min(rib_spacing,mid_length))**2
            crit_s_buckling_lower = ks_lower*E*(self.t_dcell_lower/min(rib_spacing,lower_length))**2
            crit_s_buckling_mid = ks_mid*E*(self.t_dcell_mid/min(rib_spacing,mid_length))**2
            crit_s_buckling_upper = ks_upper*E*(self.t_dcell_upper/min(rib_spacing,upper_length))**2
            crit_axial_yield = sigma_y
            crit_s_yield = sigma_y/np.sqrt(3)
            # Checks:
            # Lower
            if axial_stress_lower > min(crit_axial_yield,crit_c_buckling_lower) or shear_stress_lower > min(crit_s_buckling_lower,crit_s_yield) :
                pass_flag = False
            # Mid
            if axial_stress_mid > min(crit_axial_yield,crit_c_buckling_mid) or shear_stress_mid > min(crit_s_buckling_mid,crit_s_yield):
                pass_flag = False
            # Upper
            if axial_stress_upper > crit_axial_yield or shear_stress_upper > min(crit_s_buckling_upper,crit_s_yield):
                pass_flag = False
        return pass_flag
    
    def rib_check(self, torque, rib_material, panel_s_buckling):
        '''
        Checking main ribs and dcell ribs for shear yield and shear buckling
        Inputs:
        Torque at this section [Nm]
        Rib material, object of the material class
        Panel buckling data from ESDU 71005
        '''
        # If no rib, pass
        if self.rib_thickness == 0:
            pass_flag = True
        # If there is a rib
        else:
            # Initialising
            pass_flag = False
            # Geometry
            wingbox_height = 0.1013*self.chord
            wingbox_width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar)
            wingbox_area = wingbox_height * wingbox_width
            dcell_area = (np.pi*self.chord*self.ratio_frontSpar*wingbox_height)/2
            x_ks_mainrib = wingbox_height/wingbox_width
            ks_mainrib = panel_s_buckling.find_y(x_ks_mainrib)
            x_ks_dcellrib = wingbox_height / (self.chord*self.ratio_frontSpar)
            ks_dcellrib = panel_s_buckling.find_y(x_ks_dcellrib)
            # Material Properties
            sigma_y = rib_material.sigma_y
            E = rib_material.E
            # Applied shear stresses
            tau_applied_mainrib = np.abs(torque) / (2*(wingbox_area+dcell_area)*self.rib_thickness)
            tau_applied_dcellrib = np.abs(torque) / (2*(wingbox_area+dcell_area)*self.dcell_rib_thickness)
            # Critical shear yield stress (Von Mises)
            tau_crit_y = sigma_y / np.sqrt(3)
            # Critical shear buckling stress (ESDU 71005)
            tau_crit_b_mainrib = ks_mainrib*E*(self.rib_thickness/wingbox_height)**2
            tau_crit_b_dcellrib = ks_dcellrib*E*(self.dcell_rib_thickness/wingbox_height)**2
            # Check against smallest value
            if tau_applied_mainrib < min(tau_crit_y, tau_crit_b_mainrib):
                if tau_applied_dcellrib < min(tau_crit_y, tau_crit_b_dcellrib):
                    pass_flag = True
        return pass_flag
    
    def max_skin_stress_section(self, Mx, T):
        # Geometrical properties and coefficients
        width = self.chord * (self.ratio_rearSpar-self.ratio_frontSpar) #width of wing box
        height = self.chord * 0.1013 # Height of the box - Thickness at 15% of the chord
        stringer_spacing_upper = width/(self.n_stringers_upper+1)# Spacing between two stringers
        stringer_spacing_lower = width/(self.n_stringers_lower+1)# Spacing between two stringers
        t_eff_upper = self.t_skin_upper + self.stringer_area/stringer_spacing_upper # Effective thickness accounting for stringers - See textbook p339/340
        t_eff_lower = self.t_skin_lower + self.stringer_area/stringer_spacing_lower # Effective thickness accounting for stringers - See textbook p339/340
        # Moment of inertia of section
        I = (width*(t_eff_lower+t_eff_upper)*(height/2)**2
        + (self.t_frontSpar_web+self.t_rearSpar_web)*height**3/12
        + 4*self.spar_cap_width*self.t_Spar_cap*(height/2)**2)
        # Axial stress (same for upper and lower)
        axial_stress = np.abs(Mx)*(height/2)/I # Axial stress per unit length [N/m²] - See Textbook p339/340
        # Shear stress due to torque in lower skin - Bredt-Batho
        shear_stress_lower = np.abs(T) / (2*width*height*t_eff_lower)
        # Shear stress due to torque in upper skin - Bredt-Batho
        shear_stress_upper = np.abs(T) / (2*width*height*t_eff_upper)
        # Maximum stress
        max_stress = max(axial_stress, shear_stress_lower, shear_stress_upper)
        return max_stress