import numpy as np
import pandas as pd

# 1. Исходные данные: матрица потерь (затраты в тыс. ₸)
# Строки — стратегии (R1 - R4), столбцы — состояния природы (S1 - S4)
strategies = [
    "R1 (Пешие)",
    "R2 (Смешанный)",
    "R3 (Авто/мото)",
    "R4 (Макс. резерв)",
]
states = ["S1 (Штатный)", "S2 (Час пик)", "S3 (Выходные)", "S4 (Непогода)"]

matrix = np.array(
    [
        [50.0, 90.0, 140.0, 260.0],
        [70.0, 100.0, 110.0, 190.0],
        [120.0, 130.0, 120.0, 160.0],
        [180.0, 150.0, 135.0, 130.0],
    ]
)

df = pd.DataFrame(matrix, index=strategies, columns=states)
print("Исходная матрица потерь:")
print(df)
print("-" * 50)

# 2. Критерий Лапласа (минимизация среднего арифметического при равных вероятностях)
laplace_means = df.mean(axis=1)
opt_laplace = laplace_means.idxmin()
print("Критерий Лапласа (средние затраты):")
print(laplace_means)
print(f"-> Оптимальная стратегия: {opt_laplace}\n" + "-" * 50)

# 3. Критерий Вальда (Минимакс для потерь: минимум из максимальных затрат в строках)
max_in_rows = df.max(axis=1)
opt_wald = max_in_rows.idxmin()
print("Критерий Вальда (максимальные потери по строкам):")
print(max_in_rows)
print(f"-> Оптимальная стратегия: {opt_wald}\n" + "-" * 50)

# 4. Критерий Сэвиджа (Минимизация максимального риска / сожалений)
# Шаг 4.1: Минимальные потери в каждом столбце
min_in_cols = df.min(axis=0)
# Шаг 4.2: Матрица рисков r_ij = V_ij - min(V_j)
regret_matrix = df - min_in_cols
max_regret = regret_matrix.max(axis=1)
opt_savage = max_regret.idxmin()

print("Матрица рисков (сожалений):")
print(regret_matrix)
print("\nМаксимальный риск для каждой стратегии:")
print(max_regret)
print(f"-> Оптимальная стратегия по Сэвиджу: {opt_savage}\n" + "-" * 50)

# 5. Критерий Гурвица для альфа от 0 до 1 с шагом 0.1
# Для потерь: W_i = alpha * min_j(V_ij) + (1 - alpha) * max_j(V_ij)
min_row = df.min(axis=1)
max_row = df.max(axis=1)

hurwicz_results = []
alphas = [round(a, 1) for a in np.arange(0, 1.1, 0.1)]

for alpha in alphas:
  w = alpha * min_row + (1 - alpha) * max_row
  best_strat = w.idxmin()
  hurwicz_results.append(
      {
          "Alpha (α)": alpha,
          "Оптимальная стратегия": best_strat,
          "Минимальное W": round(w.min(), 2),
      }
  )

hurwicz_df = pd.DataFrame(hurwicz_results)
print("Результаты по критерию Гурвица при разных коэффициентах α:")
print(hurwicz_df.to_string(index=False))