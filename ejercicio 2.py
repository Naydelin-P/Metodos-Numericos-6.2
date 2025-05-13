import numpy as np
import matplotlib.pyplot as plt

# Definimos la ecuación diferencial: dq/dt = (V - q/C) / R
def f(t, q, V=10, R=1000, C=0.001):
    return (V - q/C) / R

# Método de Runge-Kutta de 4to orden con impresión de valores
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]

    t = t0
    q = q0

    # Encabezado de la tabla
    print(f"{'t':>10} {'q_aproximado':>15}")
    print(f"{t:10.4f} {q:15.6f}")

    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)

        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(t)
        q_vals.append(q)

        print(f"{t:10.4f} {q:15.6f}")

    return np.array(t_vals), np.array(q_vals)

# Parámetros iniciales
t0 = 0
q0 = 0  # Carga inicial
t_end = 1
h = 0.05

# Llamada al método de Runge-Kutta
t_vals, q_vals = runge_kutta_4(f, t0, q0, t_end, h)

# Graficar la solución
plt.figure(figsize=(8,5))
plt.plot(t_vals, q_vals, 'bo-', label="Solución RK4")
plt.xlabel("t")
plt.ylabel("q (Carga)")
plt.title("Carga de un capacitor en un circuito RC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("carga_capacitor_rk4.png")
plt.show()
