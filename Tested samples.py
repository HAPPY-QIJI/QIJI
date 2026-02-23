# FRIDAY EVENING - ALL UPPER SKINS FAILED
range_spar_material = [al_7075_T6, ti_6al_4v]
range_stringer_material = [al_2024_T3, al_7075_T6]
range_rib_material = [al_7075_T6, al_2024_T3]
range_t_skin_zones_lower = [
    [(0.2, 0.008+i*0.001), (0.5, 0.006+j*0.001), (1.0, 0.002+k*0.001)]
    for i in range(3)   # root
    for j in range(2)   # mid
    for k in range(1)   # tip
]
range_t_skin_zones_upper = [
    [(0.2, 0.008+i*0.001), (0.5, 0.006+j*0.001), (1.0, 0.002+k*0.002)]
    for i in range(3)   # root
    for j in range(2)   # mid
    for k in range(1)   # tip
]

range_stringer_area = [0.0003 + i*0.00015 for i in range(2)]

range_stringer_t_table = [
    [(0.003 + i*0.0015) for _ in range(Nsections)]
    for i in range(3)
]

range_stringer_pitch_upper = [0.10 + i*0.05 for i in range(3)]
range_stringer_pitch_lower = [0.10 + i*0.05 for i in range(3)]
range_rib_spacing_root_to_eng1 = [0.3, 0.35]
range_rib_spacing_eng1_to_eng2 = [0.35, 0.45]
range_rib_spacing_eng2_to_eng3 = [0.55]
range_rib_spacing_eng3_to_tip  = [0.60]

# FRIDAY NIGHT
# Ranges of parameters
range_spar_material = [al_7075_T6, ti_6al_4v]
range_stringer_material = [al_2024_T3, al_7075_T6]
range_rib_material = [al_7075_T6, al_2024_T3]
range_t_skin_zones_lower = [
    [(0.2, 0.008+i*0.001), (0.5, 0.005+j*0.001), (1.0, 0.002+k*0.001)]
    for i in range(3)   # root
    for j in range(3)   # mid
    for k in range(1)   # tip
]
range_t_skin_zones_upper = [
    [(0.2, 0.11+i*0.001), (0.5, 0.005+j*0.001), (1.0, 0.002+k*0.001)]
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