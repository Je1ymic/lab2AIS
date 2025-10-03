import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Генерация csv файла
def data():
    n_rows = 500
    x1_min, x1_max = -2.0, 2.0
    x2_min, x2_max = -5.0, 5.0
    x1 = np.linspace(x1_min, x1_max, n_rows)
    x2 = np.linspace(x2_min, x2_max, n_rows)

    y = x1 ** 6 + x2 ** 2 + x1 ** 3 + 4 * x2 + 5

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })

    df.to_csv('data.csv', index=False)
    print("Файл 'data.csv' успешно создан.")


# Построение графиков
def plot_graphs():
    df = pd.read_csv('data.csv')
    x1_const = df['x1'].mean()
    x2_const = df['x2'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x2_filtered = df[np.abs(df['x2'] - x2_const) < 0.1]
    if len(x2_filtered) > 0:
        ax1.plot(x2_filtered['x1'], x2_filtered['y'], alpha=0.7, color='blue')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('y')
        ax1.set_title(f'y(x1) при x2 ≈ {x2_const}')
        ax1.grid(True, alpha=0.3)

    x1_filtered = df[np.abs(df['x1'] - x1_const) < 0.1]
    if len(x1_filtered) > 0:
        ax2.plot(x1_filtered['x2'], x1_filtered['y'], alpha=0.7, color='blue')
        ax2.set_xlabel('x2')
        ax2.set_ylabel('y')
        ax2.set_title(f'y(x2) при x1 ≈ {x1_const}')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Вывод min, max, avg для каждого из столбцов
def min_max_avg():
    df = pd.read_csv('data.csv')
    print("\nСтатистика по столбцам:\n")
    for column in df.columns:
        print(f"{column}:\nСреднее:{df[column].mean():.4f}\nМинимальное:{df[column].min():.4f}\nМаксимальное:{df[column].max():.4f}\n")


# Фильтрация и сохранение в новый csv
def data_upd():
    df = pd.read_csv('data.csv')
    x1_mean = df['x1'].mean()
    x2_mean = df['x2'].mean()

    filtered_df = df[(df['x1'] < x1_mean) | (df['x2'] < x2_mean)]

    filtered_df.to_csv('filtered_data.csv', index=False)
    print(f"Файл 'filtered_data.csv' успешно создан.")


# Построение 3D графика
def plot_3d_graph():
    df = pd.read_csv('data.csv')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1_unique = np.linspace(df['x1'].min(), df['x1'].max(), 50)
    x2_unique = np.linspace(df['x2'].min(), df['x2'].max(), 50)
    x1, x2 = np.meshgrid(x1_unique, x2_unique)
    y = x1 ** 6 + x2 ** 2 + x1 ** 3 + 4 * x2 + 5

    surf = ax.plot_surface(x1, x2, y, cmap='viridis', alpha=0.8)

    ax.scatter(df['x1'], df['x2'], df['y'], color='red', alpha=0.3, s=1)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.set_title('3D график: y = x1⁶ + x2² + x1³ + 4x2 + 5')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()


def main():
    data()

    plot_graphs()

    min_max_avg()

    data_upd()

    plot_3d_graph()


if __name__ == "__main__":
    main()
