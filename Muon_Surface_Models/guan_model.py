import numpy as np

def modified_gaisser_formula(E, theta):
   
    """
    dF/dSAdE = [m^-2*s^-1*sr^-1*GeV^-1]
    """
    p_1 = 0.102573 ; p_2 = -0.068287 ; p_3 = 0.958633; p_4 = 0.0407253 ; p_5 = 0.817285

    cos_star = np.sqrt((np.cos(theta)**2 + p_1**2 + (p_2 * np.cos(theta)**p_3) + (p_4 * np.cos(theta)**p_5)) / (1 + p_1**2 + p_2 + p_4))

    E_star = E * (1 + (3.64/(E*cos_star**1.29)))

    return 10000 * 0.14 * E_star**-2.7 * ( (1/(1+((1.1*E*cos_star)/115))) + (0.054/(1+((1.1*E*cos_star)/850))) )

if __name__ == "__main__":
    pass
