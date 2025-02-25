import numpy as np

def rk45_integrate(f, t_span, y0, t_eval, args=(), tol=1e-6, h_min=1e-6):
    t0, tf = t_span
    t = t0
    y = np.array(y0, dtype=float)

    # Начальный шаг
    h = (tf - t0) / 1000

    # Коэффициенты метода РУНГЕ КУТТА
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    c7 = 1.0

    a21 = 1/5

    a31 = 3/40
    a32 = 9/40

    a41 = 44/45
    a42 = -56/15
    a43 = 32/9

    a51 = 19372/6561
    a52 = -25360/2187
    a53 = 64448/6561
    a54 = -212/729

    a61 = 9017/3168
    a62 = -355/33
    a63 = 46732/5247
    a64 = 49/176
    a65 = -5103/18656

    a71 = 35/384
    a72 = 0
    a73 = 500/1113
    a74 = 125/192
    a75 = -2187/6784
    a76 = 11/84

    # Весовые коэффициенты для 5-го порядка 
    b1 = 35/384
    b2 = 0
    b3 = 500/1113
    b4 = 125/192
    b5 = -2187/6784
    b6 = 11/84
    b7 = 0

    # Весовые коэффициенты для 4-го порядка
    b1_star = 5179/57600
    b2_star = 0
    b3_star = 7571/16695
    b4_star = 393/640
    b5_star = -92097/339200
    b6_star = 187/2100
    b7_star = 1/40

    # Списки для хранения решения
    ts = [t]
    ys = [y.copy()]

    # Текущий индекс для t_eval
    t_eval = np.array(t_eval)
    sol_t = []
    sol_y = []

    eval_idx = 0

    while t < tf:
        if t + h > tf:
            h = tf - t

        # Вычисляем этапы
        k1 = f(t, y, *args)
        k2 = f(t + c2 * h, y + h * a21 * k1, *args)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2), *args)
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), *args)
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), *args)
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), *args)
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), *args)

        # 5-й порядок (решение)
        y5 = y + h * (b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)

        # 4-й порядок (оценка)
        y4 = y + h * (b1_star * k1 + b2_star * k2 + b3_star * k3 + b4_star * k4 + b5_star * k5 + b6_star * k6 + b7_star * k7)

        # Оценка погрешности
        err = np.linalg.norm(y5 - y4, ord=np.inf)
        # Если ошибка приемлема, шаг принимается
        if err < tol:
            t_new = t + h
            y = y5
            t = t_new
            ts.append(t)
            ys.append(y.copy())
            # Сохраняем значения, попадающие в t_eval
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

        # Расчёт нового шага (если err == 0, увеличиваем шаг)
        if err == 0:
            s = 4.0
        else:
            s = 0.84 * (tol / err) ** 0.25
        h = s * h
        if h < h_min:
            h = h_min

    sol_t = np.array(sol_t)
    sol_y = np.array(sol_y).T 
    return {'t': sol_t, 'y': sol_y}
