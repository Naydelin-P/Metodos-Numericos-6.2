import numpy as np
import matplotlib.pyplot as plt

# Definición de la ecuación diferencial: dT/dx = -0.25(T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden con impresión de valores
def runge_kutta_4(f, x0, y0, x_end, h):
    x_vals = [x0]
    y_vals = [y0]

    x = x0
    y = y0

    # Encabezado de la tabla
    print(f"{'x':>10} {'T_aproximado':>15}")
    print(f"{x:10.4f} {y:15.6f}")

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        x_vals.append(x)
        y_vals.append(y)

        print(f"{x:10.4f} {y:15.6f}")

    return x_vals, y_vals

# Parámetros iniciales
x0 = 0
T0 = 100  # Temperatura inicial
x_end = 2
h = 0.1

# Llamada al método de Runge-Kutta
x_vals, T_vals = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta
T_exacta = [25 + 75 * np.exp(-0.25 * x) for x in x_vals]

# Graficar la solución
plt.figure(figsize=(8,5))
plt.plot(x_vals, T_vals, 'bo-', label="Solución RK4")
plt.plot(x_vals, T_exacta, 'r-', label="Solución Exacta")
plt.xlabel("x")
plt.ylabel("T (°C)")
plt.title("Transferencia de calor en un tubo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("temperatura_runge_kutta.png")
plt.show()
