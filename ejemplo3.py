import numpy as np
import matplotlib.pyplot as plt

# Definimos el sistema de ecuaciones
def f(t, y):
    y1, y2 = y
    dy1_dt = y2
    dy2_dt = -2*y2 - 5*y1
    return np.array([dy1_dt, dy2_dt], dtype=np.float64)  # Asegurar tipo float64

# Implementamos el método de Runge-Kutta de 4to orden
def runge_kutta_4(f, t0, y0, t_end, h):
    t_vals = [t0]
    y_vals = [y0.astype(np.float64)]  # Asegurar tipo float64 en la inicialización

    t = t0
    y = y0.astype(np.float64)  # Convertir y a float64

    print(f"{'t':>10} {'y1':>15} {'y2':>15}")
    print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")

    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(t)
        y_vals.append(y.copy())

        print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")

    return np.array(t_vals), np.array(y_vals)

# Condiciones iniciales
t0 = 0
y0 = np.array([1.0, 0.0], dtype=np.float64)  # Convertir a float64
t_end = 5
h = 0.1

# Resolvemos con Runge-Kutta
t_vals, y_vals = runge_kutta_4(f, t0, y0, t_end, h)

# Graficamos la solución
plt.figure(figsize=(8,5))
plt.plot(t_vals, y_vals[:,0], 'bo-', label="Posición (y1)")
plt.plot(t_vals, y_vals[:,1], 'ro-', label="Velocidad (y2)")
plt.xlabel("t")
plt.ylabel("y1 (Posición)_y2 (Velocidad)")
plt.title("Movimiento de un resorte amortiguado")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("resorte_amortiguado.png")
plt.show()
