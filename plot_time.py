import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DOPRI8 import dopri8_integrate

def chua_deriv(t, state, G, C1, C2, L, Ga, Gb, Bp=1.0):
    v1, v2, iL = state
    g = Gb * v1 + 0.5 * (Ga - Gb) * (np.abs(v1 + Bp) - np.abs(v1 - Bp))
    dv1dt = (G * (v2 - v1) - g) / C1
    dv2dt = (G * (v1 - v2) + iL) / C2
    diLdt = -v2 / L
    return np.array([dv1dt, dv2dt, diLdt])

parser = argparse.ArgumentParser(description="Симуляция цепи Чуа: графики v1(t) и v2(t) (DOPRI8)")
parser.add_argument('--set', default='default', help="Имя набора параметров из файла params.json")
args = parser.parse_args()

with open('params.json', 'r') as f:
    data = json.load(f)
params = data.get(args.set)
if params is None:
    raise ValueError(f"Набор параметров '{args.set}' не найден в файле params.json")
print(f"Используем набор параметров: {args.set}")
print(params)

state0 = [0.1, 0.0, 0.0]
t_span = (0, 100)
n_points = 10000
t_eval = np.linspace(t_span[0], t_span[1], n_points)

# Решение системы методом DOPRI8
sol = dopri8_integrate(
    chua_deriv, t_span, state0, t_eval,
    args=(params['G'], params['C1'], params['C2'], params['L'], params['Ga'], params['Gb']),
    tol=1e-7
)

t = sol['t']
v1 = sol['y'][0]
v2 = sol['y'][1]

# Построение графика v1(t)
plt.figure()
plt.plot(t, v1, color='blue', lw=0.5)
plt.xlabel('t')
plt.ylabel('v1 (В)')
plt.title('Зависимость v1 от времени в цепи Чуа (DOPRI8)')
plt.grid(True)

# Построение графика v2(t)
plt.figure()
plt.plot(t, v2, color='red', lw=0.5)
plt.xlabel('t')
plt.ylabel('v2 (В)')
plt.title('Зависимость v2 от времени в цепи Чуа (DOPRI8)')
plt.grid(True)

plt.show()
