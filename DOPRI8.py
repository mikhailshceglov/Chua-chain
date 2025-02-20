import numpy as np

def dopri8_integrate(f, t_span, y0, t_eval, args=(), tol=1e-6, h_max=None, h_min=1e-8):
    """
    Упрощённая (демонстрационная) реализация 8-стадийного метода, НЕ являющаяся
    классическим Dormand–Prince 8(7) с 13 стадиями. 
    Изменены b и b_star, чтобы избежать нулевой ошибки и экспоненциального роста шага.
    """
    t0, tf = t_span
    t = t0
    y = np.array(y0, dtype=float)

    # Начальный шаг
    if h_max is None:
        h = (tf - t0) / 1000
    else:
        h = min((tf - t0) / 1000, h_max)
    
    # Упрощённые коэффициенты (8 стадий)
    c = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
    A = [
        [],           # k1
        [0.1],        # k2
        [0.05, 0.15], # k3
        [0.1, 0.1, 0.1], 
        [0.05, 0.1, 0.1, 0.15],
        [0.1, 0.1, 0.1, 0.1, 0.2],
        [0.05, 0.05, 0.1, 0.1, 0.1, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ]
    # b — веса для «основного» решения
    b = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]  # сумма = 1
    # b_star — веса для «оценки ошибки» (сделаны отличными, чтобы ошибка не была 0)
    b_star = [0.14, 0.11, 0.13, 0.12, 0.12, 0.12, 0.13, 0.13]      # сумма ≈ 1, но другая

    # Подготовка к накоплению решения
    ts = [t]
    ys = [y.copy()]
    t_eval = np.array(t_eval)
    sol_t = []
    sol_y = []
    eval_idx = 0

    while t < tf:
        if t + h > tf:
            h = tf - t
        
        # Вычисляем k1...k8
        k = []
        k1 = f(t, y, *args)
        k.append(k1)
        for i in range(1, 8):
            ti = t + c[i] * h
            yi = y.copy()
            for j in range(i):
                yi += h * A[i][j] * k[j]
            ki = f(ti, yi, *args)
            k.append(ki)
        
        # Вычисляем «8-го порядка» и «7-го порядка»
        y8 = y.copy()
        y7 = y.copy()
        for i in range(8):
            y8 += h * b[i] * k[i]
            y7 += h * b_star[i] * k[i]
        
        # Оценка ошибки
        err = np.linalg.norm(y8 - y7, ord=np.inf)

        # Если ошибка приемлема, «продвигаемся» дальше
        if err < tol:
            t += h
            y = y8
            ts.append(t)
            ys.append(y.copy())
            # Сохраняем значения в моменты из t_eval, попадающие на текущий шаг
            while eval_idx < len(t_eval) and t_eval[eval_idx] <= t:
                if len(ts) >= 2:
                    t_prev, t_curr = ts[-2], ts[-1]
                    y_prev, y_curr = ys[-2], ys[-1]
                    theta = (t_eval[eval_idx] - t_prev) / (t_curr - t_prev)
                    sol_y.append(y_prev + theta * (y_curr - y_prev))
                else:
                    sol_y.append(y.copy())
                sol_t.append(t_eval[eval_idx])
                eval_idx += 1

        # Корректируем шаг (чтобы не «улетал» слишком быстро)
        if err == 0:
            s = 2.0
        else:
            s = 0.84 * (tol / err) ** (1/8)
            s = min(s, 2.0)  # не даём слишком резко увеличивать шаг
        
        h = s * h
        if h_max is not None and h > h_max:
            h = h_max
        if h < h_min:
            h = h_min

    sol_t = np.array(sol_t)
    sol_y = np.array(sol_y).T  # (n_variables, n_times)
    return {'t': sol_t, 'y': sol_y}
