from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation
import numpy as np
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from numpy.fft import fft

prach_config = PrachConfig()

prach_config.preambleIndex = 60
prach_config.prachConfigurationIndex = 158
prach_config.rootSequenceIndex = 39
prach_config.subcarrierSpacing = 30
prach_config.zeroCorrelationZoneConfig = 8
prach_config.frequencyRange = 'FR1'
prach_config.set = 'UnrestrictedSet'
prach_config.spectrumType = 'Unpaired'
prach_config.frequencyStart = '0'

carrier_config = CarrierConfig()

carrier_config.n_UL_RB = 273
carrier_config.subcarrierSpacing = 30
carrier_config.numFrame = 1

random_access_config = RandomAccessConfig()
random_access_config.get_full_random_access_config(prach_config, carrier_config)

prach_config.display_config()
carrier_config.display_config()
random_access_config.display_random_access_config()

N_CS = get_NCS(prach_config, random_access_config)
print('-----------------Sequence Configuration-----------------')
print(f"N_CS = {N_CS}")

u, u_arr_unique = get_u(prach_config, random_access_config, N_CS)
print(u)
print(u_arr_unique)

C_v, C_v_arr = get_C_v(prach_config, random_access_config, N_CS)
print(C_v)
print(C_v_arr)

x_u = calcBaseZC(random_access_config.L_RA, u)
print('----------------x_u-----------------')
print(x_u)

x_uv = np.roll(x_u, -C_v)
print('----------------x_uv-----------------')
print(x_uv)

y_uv = fft(x_uv)
print('----------------y_uv-----------------')
print(y_uv)

time_domain_signal, start_mapping_symbol, end_mapping_symbol = prach_modulation(prach_config, carrier_config, random_access_config)


print(time_domain_signal[start_mapping_symbol[0]:end_mapping_symbol[0] + 1])








