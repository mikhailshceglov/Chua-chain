import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Функция, описывающая систему ОДУ цепи Чуа
def chua_deriv(t, state, G, C1, C2, L, Ga, Gb, Bp=1.0):
    v1, v2, iL = state
    g = Gb * v1 + 0.5 * (Ga - Gb) * (np.abs(v1 + Bp) - np.abs(v1 - Bp))
    dv1dt = (G * (v2 - v1) - g) / C1
    dv2dt = (G * (v1 - v2) + iL) / C2
    diLdt = -v2 / L
    return [dv1dt, dv2dt, diLdt]

params = {
    'L': 1/7,    # индуктивность (Гн)
    'G': 0.7,    # проводимость (См)
    'C1': 1/9,   # ёмкость C1 (Ф)
    'C2': 1,     # ёмкость C2 (Ф)
    'Ga': -0.8,  # коэффициент нелинейного элемента (А/В)
    'Gb': -0.5   # коэффициент нелинейного элемента (А/В)
}

# Начальные условия
state0 = [0.1, 0.0, 0.0]

# Временной интервал моделирования
t_span = (0, 100)
n_points = 10000
t_eval = np.linspace(t_span[0], t_span[1], n_points)

# Получаем решение методом RK45
sol_RK45 = solve_ivp(chua_deriv, t_span, state0, t_eval=t_eval, args=(params['G'], params['C1'], params['C2'], params['L'], params['Ga'], params['Gb']), method='RK45')

# Получаем решение методом DOP853 (Dormand-Prince 8(5,3))
sol_DOP853 = solve_ivp(chua_deriv, t_span, state0, t_eval=t_eval, args=(params['G'], params['C1'], params['C2'], params['L'], params['Ga'], params['Gb']), method='DOP853')

# Извлекаем переменные: v_C1 и i_L для обоих методов
v1_RK45, iL_RK45 = sol_RK45.y[0], sol_RK45.y[2]
v1_DOP853, iL_DOP853 = sol_DOP853.y[0], sol_DOP853.y[2]
all_v1 = np.concatenate((v1_RK45, v1_DOP853))
all_iL = np.concatenate((iL_RK45, iL_DOP853))
xlim = (np.min(all_v1), np.max(all_v1))
ylim = (np.min(all_iL), np.max(all_iL))

# Создаем фигуру с 4 подграфиками (2 строки, 2 столбца)
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Подграфик 1: Анимация RK45
ax1 = axs[0, 0]
ax1.set_title("Анимация: RK45")
ax1.set_xlabel("v_C1 (В)")
ax1.set_ylabel("i_L (А)")
ax1.grid(True)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
line_RK45_ax1, = ax1.plot([], [], lw=0.5, color='blue', label='RK45')
ax1.legend()

# Подграфик 2: Анимация DOP853
ax2 = axs[0, 1]
ax2.set_title("Анимация: DOP853")
ax2.set_xlabel("v_C1 (В)")
ax2.set_ylabel("i_L (А)")
ax2.grid(True)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
line_DOP853_ax2, = ax2.plot([], [], lw=0.5, color='magenta', label='DOP853')
ax2.legend()

# Подграфик 3: Совместная анимация (RK45 и DOP853)
ax3 = axs[1, 0]
ax3.set_title("Совместная анимация: RK45 и DOP853")
ax3.set_xlabel("v_C1 (В)")
ax3.set_ylabel("i_L (А)")
ax3.grid(True)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
line_RK45_ax3, = ax3.plot([], [], lw=0.5, color='blue', label='RK45')
line_DOP853_ax3, = ax3.plot([], [], lw=0.5, color='magenta', label='DOP853')
ax3.legend()

# Подграфик 4: Статический итоговый график
ax4 = axs[1, 1]
ax4.set_title("Итоговый фазовый портрет")
ax4.set_xlabel("v_C1 (В)")
ax4.set_ylabel("i_L (А)")
ax4.grid(True)
ax4.set_xlim(xlim)
ax4.set_ylim(ylim)
# Статически отображаем полные траектории
ax4.plot(v1_RK45, iL_RK45, lw=0.5, color='blue', label='RK45')
ax4.plot(v1_DOP853, iL_DOP853, lw=0.5, color='magenta', label='DOP853')
ax4.legend()

# Функция инициализации для анимации (обновляются 3 подграфика)
def init():
    line_RK45_ax1.set_data([], [])
    line_DOP853_ax2.set_data([], [])
    line_RK45_ax3.set_data([], [])
    line_DOP853_ax3.set_data([], [])
    return line_RK45_ax1, line_DOP853_ax2, line_RK45_ax3, line_DOP853_ax3

# Функция обновления анимации для первых трех подграфиков
def update(frame):
    line_RK45_ax1.set_data(v1_RK45[:frame], iL_RK45[:frame])
    line_DOP853_ax2.set_data(v1_DOP853[:frame], iL_DOP853[:frame])
    line_RK45_ax3.set_data(v1_RK45[:frame], iL_RK45[:frame])
    line_DOP853_ax3.set_data(v1_DOP853[:frame], iL_DOP853[:frame])
    return line_RK45_ax1, line_DOP853_ax2, line_RK45_ax3, line_DOP853_ax3

# Создаем анимацию для первых 3 подграфиков
anim = FuncAnimation(fig, update, frames=n_points, init_func=init, interval=1, blit=True)

plt.tight_layout()
plt.show()
