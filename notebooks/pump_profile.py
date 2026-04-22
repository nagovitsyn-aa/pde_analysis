import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import quad
    return mo, np, plt, quad


@app.cell
def _(np, quad):
    # Определяем гауссиану
    def P_g(z):
        return np.exp(-z**2 / 2)
    
    # Вычисляем интеграл гауссианы от -∞ до +∞
    integral_g, _ = quad(P_g, -np.inf, np.inf)
    
    # Для ступеньки: нужно найти границы A и B такие, что 
    # площадь ступеньки высотой 1 от A до B = integral_g
    # Положим симметричные границы: от -L до L
    # Площадь ступеньки = 2L
    # Тогда 2L = integral_g → L = integral_g / 2
    L_step = integral_g / 2
    
    def step_function(z):
        return 1.0 if abs(z) <= L_step else 0.0
    
    # Для параболы с подбором границ: функция 1 - z^2/2, обрезанная на границах ±L_parab
    # Нужно найти L_parab такое, что интеграл от -L_parab до L_parab от (1 - z^2/2) dz = integral_g
    # ∫ (1 - z^2/2) dz = z - z^3/6
    # От -L до L: (L - L^3/6) - (-L + L^3/6) = 2L - L^3/3
    # Решаем 2L - L^3/3 = integral_g
    # Решим численно
    def integral_parabola(L):
        return 2*L - L**3/3
    
    # Бинарный поиск для нахождения L
    L_parab = integral_g / 2  # начальное приближение
    for _ in range(50):
        int_val = integral_parabola(L_parab)
        if int_val < integral_g:
            L_parab *= 1.02
        else:
            L_parab *= 0.98
    
    def parabola_cut(z):
        if abs(z) <= L_parab:
            return 1 - z**2 / 2
        else:
            return 0.0
    
    # Классическая парабола с естественными границами z = ±√2
    L_classic = np.sqrt(2)
    integral_classic, _ = quad(lambda z: 1 - z**2/2, -L_classic, L_classic)
    
    def parabola_classic(z):
        if abs(z) <= L_classic:
            return 1 - z**2 / 2
        else:
            return 0.0
    
    # Подготовим данные для вывода
    mo.md(f"""
    **Интегралы функций:**
    - Гауссиана ∫ P_g(z) dz = {integral_g:.4f}
    - Ступенчатая функция (ширина 2×{L_step:.3f}) ∫ = {integral_parabola(L_step):.4f}
    - Парабола с подбором границ (±{L_parab:.3f}) ∫ = {integral_parabola(L_parab):.4f}
    - Классическая парабола (±{L_classic:.3f}) ∫ = {integral_classic:.4f}
    """)
    
    return (
        L_classic,
        L_parab,
        L_step,
        P_g,
        integral_classic,
        integral_g,
        integral_parabola,
        parabola_classic,
        parabola_cut,
        step_function,
    )


@app.cell
def _(
    L_classic,
    L_parab,
    L_step,
    P_g,
    integral_classic,
    integral_g,
    parabola_classic,
    parabola_cut,
    plt,
    step_function,
):
    # Создаём сетку по z
    z = np.linspace(-5, 5, 500)
    
    # Вычисляем значения функций
    y_g = P_g(z)
    y_step = [step_function(zi) for zi in z]
    y_parab_cut = [parabola_cut(zi) for zi in z]
    y_parab_classic = [parabola_classic(zi) for zi in z]
    
    # Рисуем график
    plt.figure(figsize=(12, 7))
    
    plt.plot(z, y_g, linewidth=2.5, 
             label=f'$P_g(z) = e^{{-z^2/2}}$ [∫ = {integral_g:.3f}]')
    
    plt.plot(z, y_step, '--', linewidth=2, 
             label=f'Step function [∫ = {integral_g:.3f}], |z| ≤ {L_step:.3f}')
    
    plt.plot(z, y_parab_cut, ':', linewidth=2, 
             label=f'Parabola $1 - z^2/2$, tuned bounds [∫ = {integral_g:.3f}], |z| ≤ {L_parab:.3f}')
    
    plt.plot(z, y_parab_classic, '-.', linewidth=2, 
             label=f'Parabola $1 - z^2/2$, classic bounds [∫ = {integral_classic:.3f}], |z| ≤ {L_classic:.3f}')
    
    plt.xlabel('z', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title('Comparison of functions with their integrals in legend', fontsize=14)
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.2, 1.2)
    
    plt.gca()
    return y_g, y_parab_classic, y_parab_cut, y_step, z


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()