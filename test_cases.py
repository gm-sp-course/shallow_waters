import firedrake
import numpy

# Physical constants ----------------------------------------------------------
g = 9.80616
# -----------------------------------------------------------------------------

# Gaussian hill ---------------------------------------------------------------
def h_0_lambda_gaussian_hill(x, y):
    sigma = 0.05
    amplitude = 0.01
    Lx = 1.0  # gaussian_hill_default_parameters["Lx"]
    Ly = 1.0  # gaussian_hill_default_parameters["Ly"]

    return firedrake.Constant(1.0) + amplitude*firedrake.exp(-0.5*((x - 0.5*Lx)**2)/sigma**2)*(1.0/(sigma*firedrake.sqrt(2.0*numpy.pi))) * firedrake.exp(-0.5*((y - 0.5*Ly)**2)/sigma**2)*(1.0/(sigma*firedrake.sqrt(2.0*numpy.pi)))

def u_0_lambda_gaussian_hill(x, y):
    return firedrake.as_vector([firedrake.Constant(0.0), firedrake.Constant(0.0)])

gaussian_hill_default_parameters = {"Lx": 1.0, \
                                    "Ly": 1.0, \
                                    "n_elements_x": 25, \
                                    "n_elements_y": 25, \
                                    "p": 3, \
                                    "dt": 0.01, \
                                    "n_t_steps": 50, \
                                    "h_0_lambda": h_0_lambda_gaussian_hill, \
                                    "u_0_lambda": u_0_lambda_gaussian_hill}

# -----------------------------------------------------------------------------


# Travelling wave -------------------------------------------------------------
def h_0_lambda_travelling_wave(x, y):
    sigma = 0.05
    amplitude = 0.01
    Lx = 4.0  # travelling_wave_default_parameters["Lx"]

    return firedrake.Constant(1.0) + amplitude*firedrake.exp(-0.5*((x - 0.5*Lx)**2)/sigma**2)*(1.0/(sigma*firedrake.sqrt(2.0*numpy.pi)))

def u_0_lambda_travelling_wave(x, y):
    return firedrake.as_vector([firedrake.Constant(0.0), firedrake.Constant(0.0)])

travelling_wave_default_parameters = {"Lx": 4.0, \
                                      "Ly": 0.2, \
                                      "n_elements_x": 100, \
                                      "n_elements_y": 3, \
                                      "p": 3, \
                                      "dt": 0.0025, \
                                      "n_t_steps": 800, \
                                      "h_0_lambda": h_0_lambda_travelling_wave, \
                                      "u_0_lambda": u_0_lambda_travelling_wave}

# -----------------------------------------------------------------------------


# Double vortex ---------------------------------------------------------------
def h_0_lambda_double_vortex(x, y):
    L = 50.0  # equal to Lx and Ly
    H_0 = firedrake.Constant(7.50)  # was 750.0
    dh = 0.75
    sigma_x = (3.0/40.0) * L
    sigma_y = (3.0/40.0) * L
    ox = 0.1
    oy = 0.1

    xc_1 = (0.5 - ox) * L
    yc_1 = (0.5 - oy) * L

    xc_2 = (0.5 + ox) * L
    yc_2 = (0.5 + oy) * L

    return H_0 - dh * (firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_1)))**2) + \
                                            (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_1)))**2) )) + \
                       firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_2)))**2) + \
                                            (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_2)))**2) )) - \
                       ((4.0*numpy.pi * sigma_x * sigma_y)/(L**2)))

def u_0_lambda_double_vortex(x, y):
    L = 50.0  # equal to Lx and Ly was 5000.0
    dh = 0.750  # was 75
    sigma_x = (3.0/40.0) * L
    sigma_y = (3.0/40.0) * L
    ox = 0.1
    oy = 0.1
    f = 0.6147 # 0.00006147

    xc_1 = (0.5 - ox) * L
    yc_1 = (0.5 - oy) * L

    xc_2 = (0.5 + ox) * L
    yc_2 = (0.5 + oy) * L

    return firedrake.as_vector([-((g * dh)/(f * sigma_y)) * (\
                                ((((L)/(2.0*numpy.pi*sigma_y)) * firedrake.sin(((2.0*numpy.pi)/(L))*(y - yc_1))) * \
                                firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_1)))**2) + \
                                                     (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_1)))**2) ))) + \
                                ((((L)/(2.0*numpy.pi*sigma_y)) * firedrake.sin(((2.0*numpy.pi)/(L))*(y - yc_2))) * \
                                firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_2)))**2) + \
                                                     (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_2)))**2) ))))\
                                , \
                                ((g * dh)/(f * sigma_x)) * (\
                                ((((L)/(2.0*numpy.pi*sigma_x)) * firedrake.sin(((2.0*numpy.pi)/(L))*(x - xc_1))) * \
                                firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_1)))**2) + \
                                                     (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_1)))**2) ))) + \
                                ((((L)/(2.0*numpy.pi*sigma_y)) * firedrake.sin(((2.0*numpy.pi)/(L))*(x - xc_2))) * \
                                firedrake.exp(-0.5*( (((L/(numpy.pi*sigma_x))*firedrake.sin((numpy.pi/L)*(x - xc_2)))**2) + \
                                                     (((L/(numpy.pi*sigma_y))*firedrake.sin((numpy.pi/L)*(y - yc_2)))**2) ))))\
                                ])

double_vortex_default_parameters = {"Lx": 50.0, \
                                    "Ly": 50.0, \
                                    "n_elements_x": 50, \
                                    "n_elements_y": 50, \
                                    "p": 3, \
                                    "dt": 1.0, \
                                    "n_t_steps": 50, \
                                    "h_0_lambda": h_0_lambda_double_vortex, \
                                    "u_0_lambda": u_0_lambda_double_vortex}

# -----------------------------------------------------------------------------


# Store all default parameters ------------------------------------------------
get_test_case_default_parameters = {"gaussian_hill": gaussian_hill_default_parameters, \
                                    "travelling_wave": travelling_wave_default_parameters, \
                                    "double_vortex": double_vortex_default_parameters}
# -----------------------------------------------------------------------------