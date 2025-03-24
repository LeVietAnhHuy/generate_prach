from configuration import PrachConfig, CarrierConfig
from get_random_access_configuration import RandomAccessConfig
from get_ncs_root_cv import get_NCS, get_u, get_C_v
from prach_modulation_demodulation import prach_modulation
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from pyphysim.reference_signals.zadoffchu import calcBaseZC
from pyphysim.channels.fading import TdlChannel, TdlChannelProfile
from pyphysim.channels.fading_generators import JakesSampleGenerator
from commpy.channels import awgn


prach_config = PrachConfig()

prach_config.preambleIndex = 60
prach_config.prachConfigurationIndex = 158
prach_config.rootSequenceIndex = 39
prach_config.subcarrierSpacing = 30
prach_config.zeroCorrelationZoneConfig = 8
prach_config.frequencyRange = 'FR1'
prach_config.set = 'UnrestrictedSet'
prach_config.spectrumType = 'Unpaired'
prach_config.frequencyStart = 0

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

time_domain_signal, start_mapping_symbol, end_mapping_symbol = prach_modulation(prach_config, carrier_config,
                                                                                    random_access_config)

bandwidth = 100e6  # in Hetz
Fd = 100  # Doppler frequency (in Hz)
# Ts = 100  # Sampling interval
Ts = time_domain_signal.size  # Sampling interval

tap_powers_dB = np.array([-6.9, 0, -7.7, -2.5, -2.4, -9.9, -8.0, -6.6, -7.1, -13.0, -14.2, -16.0])

tap_delays = np.array([0, 65, 70, 190, 195, 200, 240, 325, 520, 1045, 1510, 2595])

num_rx_antennas = 1

jakesObj = JakesSampleGenerator(Fd, Ts, L=np.size(tap_powers_dB))
tdlChanlelProfile = TdlChannelProfile(tap_powers_dB=tap_powers_dB, tap_delays=tap_delays)
tdlChannel = TdlChannel(jakesObj, channel_profile=tdlChanlelProfile)

tdlChannel.set_num_antennas(num_rx_antennas=num_rx_antennas, num_tx_antennas=1)

print(f"tdlChanlelProfile.rms_delay_spread = {tdlChanlelProfile.rms_delay_spread}\n")

snr_dB = -44
num_sample = 1
received_test_signal_arr = []
for sample_index in range(num_sample):
    received_test_signal = awgn(time_domain_signal, snr_dB=snr_dB)
    received_test_signal = tdlChannel.corrupt_data(received_test_signal)
    received_test_signal_arr.append(received_test_signal)

# fig, axs = plt.subplots(num_rx_antennas + 1, 1)
fig, axs = plt.subplots(1, num_rx_antennas + 1)
axs[0].plot(time_domain_signal)
for axs_idx in range(1, received_test_signal.shape[0] + 1):
    axs[axs_idx].plot(received_test_signal[axs_idx - 1, :])
plt.show()

print('')













