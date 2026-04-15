
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
inference_couplings = np.array([0.827, 1.151, 1.514, 2.16, 1.3])

# Plot widths vs (formula couplings) and (inference couplings)
plt.figure(figsize=(10, 6))
plt.plot(widths_percent, formula_couplings, marker='o', label='From model formula')
plt.plot(widths_percent, inference_couplings, marker='s', label='From inference')
plt.xlabel('S width (%)')
plt.ylabel('Couplings')
plt.legend()
plt.savefig('widths_vs_couplings.pdf')