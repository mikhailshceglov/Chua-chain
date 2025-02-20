import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from RK45 import rk45_integrate
from DOPRI8 import dopri8_integrate

def chua_deriv(t, state, G, C1, C2, L, Ga, Gb, Bp=1.0):
    v1, v2, iL = state
    g = Gb * v1 + 0.5 * (Ga - Gb) * (np.abs(v1 + Bp) - np.abs(v1 - Bp))
    dv1dt = (G * (v2 - v1) - g) / C1
    dv2dt = (G * (v1 - v2) + iL) / C2
    diLdt = -v2 / L
    return np.array([dv1dt, dv2dt, diLdt])

# Параметры
params = {
    'L': 1/7,
    'G': 0.7,
    'C1': 1/9,
    'C2': 1,
    'Ga': -0.8,
    'Gb': -0.5
}

state0 = [0.1, 0.0, 0.0]
t_span = (0, 100)
n_points = 10000
t_eval = np.linspace(t_span[0], t_span[1], n_points)

# Решения
sol_RK45 = rk45_integrate(chua_deriv, t_span, state0, t_eval,args=(params['G'], params['C1'], params['C2'], params['L'], params['Ga'], params['Gb']), tol=1e-7)

sol_DOPRI8 = dopri8_integrate(chua_deriv, t_span, state0, t_eval, args=(params['G'], params['C1'], params['C2'], params['L'], params['Ga'], params['Gb']), tol=1e-7)

v1_RK45 = sol_RK45['y'][0]
iL_RK45 = sol_RK45['y'][2]
v1_DOPRI8 = sol_DOPRI8['y'][0]
iL_DOPRI8 = sol_DOPRI8['y'][2]

# Границы графиков
all_v1 = np.concatenate((v1_RK45, v1_DOPRI8))
all_iL = np.concatenate((iL_RK45, iL_DOPRI8))
xlim = (np.min(all_v1), np.max(all_v1))
ylim = (np.min(all_iL), np.max(all_iL))

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Подграфик 1: анимация RK45
ax1 = axs[0, 0]
ax1.set_title("Анимация: RK45 (Custom)")
ax1.set_xlabel("v_C1 (В)")
ax1.set_ylabel("i_L (А)")
ax1.grid(True)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
line_RK45_ax1, = ax1.plot([], [], lw=0.5, color='blue', label='RK45 (Custom)')
ax1.legend()

# Подграфик 2: анимация DOPRI8
ax2 = axs[0, 1]
ax2.set_title("Анимация: DOPRI8 (Custom)")
ax2.set_xlabel("v_C1 (В)")
ax2.set_ylabel("i_L (А)")
ax2.grid(True)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
line_DOPRI8_ax2, = ax2.plot([], [], lw=0.5, color='magenta', label='DOPRI8 (Custom)')
ax2.legend()

# Подграфик 3: совместная анимация
ax3 = axs[1, 0]
ax3.set_title("Совместная анимация: RK45 и DOPRI8")
ax3.set_xlabel("v_C1 (В)")
ax3.set_ylabel("i_L (А)")
ax3.grid(True)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
line_RK45_ax3, = ax3.plot([], [], lw=0.5, color='blue', label='RK45 (Custom)')
line_DOPRI8_ax3, = ax3.plot([], [], lw=0.5, color='magenta', label='DOPRI8 (Custom)')
ax3.legend()

# Подграфик 4: итоговый фазовый портрет
ax4 = axs[1, 1]
ax4.set_title("Итоговый фазовый портрет")
ax4.set_xlabel("v_C1 (В)")
ax4.set_ylabel("i_L (А)")
ax4.grid(True)
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)
ax4.plot(v1_RK45, iL_RK45, lw=0.5, color='blue', label='RK45 (Custom)')
ax4.plot(v1_DOPRI8, iL_DOPRI8, lw=0.5, color='magenta', label='DOPRI8 (Custom)')
ax4.legend()

# Анимация
def init():
    line_RK45_ax1.set_data([], [])
    line_DOPRI8_ax2.set_data([], [])
    line_RK45_ax3.set_data([], [])
    line_DOPRI8_ax3.set_data([], [])
    return line_RK45_ax1, line_DOPRI8_ax2, line_RK45_ax3, line_DOPRI8_ax3

def update(frame):
    line_RK45_ax1.set_data(v1_RK45[:frame], iL_RK45[:frame])
    line_DOPRI8_ax2.set_data(v1_DOPRI8[:frame], iL_DOPRI8[:frame])
    line_RK45_ax3.set_data(v1_RK45[:frame], iL_RK45[:frame])
    line_DOPRI8_ax3.set_data(v1_DOPRI8[:frame], iL_DOPRI8[:frame])
    return line_RK45_ax1, line_DOPRI8_ax2, line_RK45_ax3, line_DOPRI8_ax3

anim = FuncAnimation(fig, update, frames=n_points, init_func=init,
                     interval=1, blit=True)

plt.tight_layout()
plt.show()
