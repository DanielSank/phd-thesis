import numpy as np

K_SAPPHIRE = 10.4
LIGHT_SPEED = 3.0E8
K_EFF = (1 + K_SAPPHIRE) / 2.0
C_q = 85.0E-15

def Cc(omega, Qc, C, Re):
    """
    Coupling capacitor needed to acheive a specified Qc
    
    This is for a series capacitor/resistor shunt connected to a resonant mode.
    """
    return (C / (Qc * omega * Re))**0.5

def CLambda4(omega, Z0):
    """Effective capacitance of lambda/4 resonator"""
    return np.pi / (4.0 * omega * Z0)

def ZLambda4(Z0):
    return (4/np.pi)*Z0

def CcLambda4(omega, Qc, Z0=50.0):
    """
    Coupling capacitor need to acheive specified Qc for a lambda/4 resonator
    
    This is for the case of a series capacitor/resistor shunt attached to the
    voltage antinode of the lambda/4 resonator.
    """
    C = CLambda4(omega, Z0)
    return Cc(omega, Qc, C, Z0)

def loadImpedance(C, omega, Z0=50.0):
    """Impedance of a series capacitor/resistor load"""
    return Z0 + 1.0/(1.0j*omega*C)

def loadLength(omega, Cin, Z0=50):
    """Electrical length of a series capacitor/resistor load"""
    ZL = loadImpedance(Cin, omega, Z0=Z0)
    gamma = (ZL - Z0) / (ZL + Z0)
    v = LIGHT_SPEED / np.sqrt(K_EFF)
    return (v/(2.0*omega)) * np.angle(gamma)

def couplingParameters(omega_r, omega_q, kappa_r, eta):
    """
    All parameters in this function are in angular frequency.
    """
    Q_r = omega_r / kappa_r
    Delta = omega_q - omega_r
    chi = -omega_r / (2 * Q_r)
    g = np.sqrt(chi * Delta * (1 + (Delta/eta)))
    return chi, g

def buildQubit(omega_r, omega_q, Q_F, Q_r, eta, Z0=50.0):
    kappa_r = omega_r / Q_r
    Z_F = ZLambda4(Z0)
    Z_r = ZLambda4(Z0)
    C_r = CLambda4(omega_r, Z0=Z0)
    
    chi, g = couplingParameters(omega_r, omega_q, kappa_r, eta)
    C_g = 2*g*np.sqrt(C_r*C_q)/np.sqrt(omega_r*omega_q)
    
    C_kappa = np.sqrt(C_r / (omega_r * Q_F * Q_r * Z_F))
    
    print("chi: %f MHz"%(chi/1.0E6/(np.pi*2),))
    print("g: %f MHz"%(g/1.0E6/(np.pi*2),))
    print("C_g: %f fF"%(C_g*1.0E15,))
    print("Q_r: %f"%(Q_r,))
    print("kappa_r^-1: %f"%((kappa_r**-1)*1E9,))
    print("C_kappa: %f fF"%(C_kappa*1.0E15,))
