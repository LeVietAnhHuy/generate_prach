import json
import numpy as np
from functools import reduce

with open('./table6_prach.txt', 'r') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'French']}
print(data['Table6.3.3.2-3']['PreambleFormat'][0])

L_RA_index = np.where(np.array(data['Table6.3.3.2-1']['L_RA']) == 139)[0]
delta_f_RA_forPrach_index = np.where(np.array(data['Table6.3.3.2-1']['delta_f_RA_forPrach']) == 30)[0]
delta_f_forPusch_index = np.where(np.array(data['Table6.3.3.2-1']['delta_f_forPusch']) == 30)[0]

print(L_RA_index)
print(delta_f_RA_forPrach_index)
print(delta_f_forPusch_index)

k_bar_index = reduce(np.intersect1d, (L_RA_index, delta_f_RA_forPrach_index, delta_f_forPusch_index))

print(k_bar_index)