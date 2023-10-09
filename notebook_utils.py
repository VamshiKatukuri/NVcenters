
import sys

sys.path.append("../nvmodels/src")
import qutip
import nvmodels
import numpy as np
import matplotlib.pyplot as plt


def zeeman_split_single_NV(B=0, lorentz_width=10e6, hyperfine=False):

    
    
    nv = nvmodels.NVNegativeGroundState()
    
    polar = 0
    azimuthal = 0
    _x = np.sin(polar)*np.cos(azimuthal)
    _y = np.sin(polar)*np.sin(azimuthal)
    _z = np.cos(polar)
    B_mag = B #100e-4  # 100 Gauss in the z direction
    static_B_field = B_mag*np.array([_x, _y, _z])
    print(static_B_field)

    h = nv.zero_field_hamiltonian()
    h += nv.nitrogen_hyperfine_hamiltonian()
    h += nv.nitrogen_electric_quadrupole_hamiltonian()
    h += nv.static_mag_field_hamiltonian(static_B_field, include_nucleus = True)
    
    eigenvalues, eigenstates = h.eigenstates()
    eigenvalues = eigenvalues - np.min(eigenvalues)
    state_probs_text = [nvmodels.utilities.two_qutrit_state_to_text(s, decimals=5) for s in eigenstates]
    
    energy_transitions = nv.electron_spin_resonances(h)
    contrast_amp = -0.01
    
    
    if hyperfine == False :
        rf_freq = np.arange(2.0e9, 3.5e9, 1e6)
        contrast = 1 + np.array([nvmodels.utilities.lorentzian(rf_freq, e, contrast_amp, lorentz_width) for e in energy_transitions]).sum(axis=0)
    else :
        rf_freq = np.arange(2.586e9, 2.594e9, .05e6)
        contrast = 1 + np.array([nvmodels.utilities.lorentzian(rf_freq, e, -.1, lorentz_width) for e in energy_transitions]).sum(axis=0)
    #lorentz_width = 30e6
    
    fig,ax = plt.subplots(figsize=(10, 5))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.set_xlabel('MW Freq. (GHz)',fontsize=20)
    ax.set_ylabel('Intensity (arb. u.)',fontsize=20)
    ax.plot(rf_freq/10e9, contrast)
    
    return None 
    
def zeeman_split_NV_ensemble(B=0,lorentz_width = 10e6):
 
    
    nv = nvmodels.NVNegativeGroundState()
    
    z = np.array([0,0,1])
    nv111 = np.array([1,1,1])/np.linalg.norm(np.array([1,1,1]))
    nv1b1b1 = np.array([1,-1,-1])/np.linalg.norm(np.array([1,-1,-1]))
    nvb11b1 = np.array([-1, 1, -1])/np.linalg.norm(np.array([-1,1,-1]))
    nvb1b11 = np.array([-1, -1, 1])/np.linalg.norm(np.array([-1,-1,1]))

    nv_orientations = [nv111, nv1b1b1, nvb11b1, nvb1b11]
    B_nv111_axis_in_lab_frame = B*np.array([1,1,1])/np.linalg.norm(np.array([1,1,1]))
    
    
    
    energy_transitions = []
    contrasts = []
    rf_freq = np.arange(2.0e9, 3.5e9, 1e6)
    #lorentz_width = 10e6
    contrast_amp = -0.01

    for nv_orient in nv_orientations:
        h = nv.zero_field_hamiltonian()
        h += nv.nitrogen_hyperfine_hamiltonian()
        h += nv.nitrogen_electric_quadrupole_hamiltonian()

        B = nvmodels.utilities.lab_to_nv_orientation(B_nv111_axis_in_lab_frame, nv_orient)

        h += nv.static_mag_field_hamiltonian(B, include_nucleus = True)
        energy_transitions += nv.electron_spin_resonances(h)
        contrasts.append(np.array([nvmodels.utilities.lorentzian(rf_freq, e, contrast_amp, lorentz_width) for e in energy_transitions]).sum(axis=0))
    
    full_spectrum = 1 + np.array(contrasts).sum(axis=0)
    
    fig,ax = plt.subplots(figsize=(10, 5))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.set_xlabel('MW Freq. (GHz)',fontsize=20)
    ax.set_ylabel('Intensity (arb. u.)',fontsize=20)
    ax.plot(rf_freq/10e9, full_spectrum)
   
    
    
def zeeman_split_NV_ensemble_B_random(B=0, lorentz_width = 10e6):
    
    
    nv = nvmodels.NVNegativeGroundState()
    
    energy_transitions = []
    contrasts = []
    # high res
    rf_freq = np.arange(2.0e9, 4.0e9, .025e6)
#    lorentz_width = .5e6
    contrast_amp = -0.01

    # low res
    # rf_freq = np.arange(2.0e9, 3.5e9, 1e6)
    # lorentz_width = 10e6
    # contrast_amp = -0.01
    
    
    
    z = np.array([0,0,1])
    nv111 = np.array([1,1,1])/np.linalg.norm(np.array([1,1,1]))
    nv1b1b1 = np.array([1,-1,-1])/np.linalg.norm(np.array([1,-1,-1]))
    nvb11b1 = np.array([-1, 1, -1])/np.linalg.norm(np.array([-1,1,-1]))
    nvb1b11 = np.array([-1, -1, 1])/np.linalg.norm(np.array([-1,-1,1]))

    nv_orientations = [nv111, nv1b1b1, nvb11b1, nvb1b11]

    v = np.array([.2,-.5,1]) #choose the direction of the B field

    B_in_lab_frame = B*v/np.linalg.norm(v)

    for nv_orient in nv_orientations:
        h = nv.zero_field_hamiltonian()
        h += nv.nitrogen_hyperfine_hamiltonian()
        h += nv.nitrogen_electric_quadrupole_hamiltonian()

        B = nvmodels.utilities.lab_to_nv_orientation(B_in_lab_frame, nv_orient)

        h += nv.static_mag_field_hamiltonian(B, include_nucleus = True)
        energy_transitions += nv.electron_spin_resonances(h)
        contrasts.append(np.array([nvmodels.utilities.lorentzian(rf_freq, e, contrast_amp, lorentz_width) for e in energy_transitions]).sum(axis=0))

    full_spectrum = 1 + np.array(contrasts).sum(axis=0)
    
    
    
    fig,ax = plt.subplots(figsize=(10, 5))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.set_xlabel('MW Freq. (GHz)',fontsize=20)
    ax.set_ylabel('Intensity (arb. u.)',fontsize=20)
    ax.plot(rf_freq/10e9, full_spectrum)