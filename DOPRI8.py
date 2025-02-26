import numpy as np

def dopri8_integrate(f, t_span, y0, t_eval, args=(), tol=1e-6, h_min=1e-8):
    t0, tf = t_span  # начало и конец отрезка интегрирования
    t = t0
    y = np.array(y0, dtype=float)  # начальное состояние

    # Начальный шаг
    h = (tf - t0) / 10000

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
    
    # b — веса для основного решения (сумма = 1)
    b = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125] # (сумма = 1)

    # b_star — веса для оценки ошибки
    b_star = [0.14, 0.11, 0.13, 0.12, 0.12, 0.12, 0.13, 0.13]    # сумма = 1, но другая

    # Подготовка к накоплению решения
    ts = [t]      # времена интегрирования
    ys = [y.copy()]  # результаты интегрирования
    t_eval = np.array(t_eval)
    sol_t = []
    sol_y = []
    eval_idx = 0

    while t < tf:
        if t + h > tf:
            h = tf - t
        
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
        
        # Вычисляем приближения 8 порядка и 7 порядка
        y8 = y.copy()
        y7 = y.copy()
        for i in range(8):
            y8 += h * b[i] * k[i]
            y7 += h * b_star[i] * k[i]
        
        # Оценка локальной ошибки
        err = np.linalg.norm(y8 - y7, ord=np.inf)

        # Если ошибка приемлема, переходим к следующему шагу
        if err < tol:
            t += h
            y = y8
            ts.append(t)
            ys.append(y.copy())
            # Сохраняем значения, попадающие в моменты из t_eval
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

        s = (tol / err) ** (1/8)

        h = s * h
        if h < h_min:
            h = h_min

    sol_t = np.array(sol_t)
    sol_y = np.array(sol_y).T
    return {'t': sol_t, 'y': sol_y}
