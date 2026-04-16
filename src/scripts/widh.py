
MT = 172
MHH = 950
couptodd = 0
coupteven = 0.7
vev = 246
import cmath

model_width = ((3*MT**2)*(MHH**2 - 4*MT**2)**0.5*((coupteven**2 + couptodd**2)*MHH**2 - (2*coupteven*MT)**2)/((8*MHH**2)*cmath.pi*vev**2))

print("Width : ", model_width)

import matplotlib.pyplot as plt
import numpy as np

widths = np.array([22.0, 47.5, 95.0, 142.5, 285.0])
widths_percent = 100 * widths / MHH
formula_couplings = coupteven * np.sqrt(widths_percent / widths_percent[0])
inference_couplings = np.array([0.863, 1.168, 1.524, 1.927, 3.0])

# Plot widths vs (formula couplings) and (inference couplings)
plt.rcParams.update({"font.size": 16})
plt.figure(figsize=(10, 6))
plt.plot(widths_percent, formula_couplings, marker='o', label='From analytical formula')
plt.plot(widths_percent, inference_couplings, marker='s', label='From inference')
plt.xlabel(r'$\frac{\Gamma(S)}{m(S)}$ (%)', fontsize=20)
plt.ylabel(r'$C_e$', fontsize=18)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('widths_vs_couplings.pdf')