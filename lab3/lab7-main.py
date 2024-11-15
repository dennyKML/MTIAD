import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
from tqdm import tqdm


# Функція для зчитування зображення у форматі RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image.astype(np.uint8)


# Функція для витягування каналу яскравості Y
def extract_brightness_channel(image):
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return ycrcb_image[:, :, 0]


# Функція для поділу на сегменти
def segment_image(image, segment_size):
    segments = []
    h, w = image.shape
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            segment = image[i:i + segment_size, j:j + segment_size]
            if segment.shape[0] == segment_size and segment.shape[1] == segment_size:
                segments.append((segment, i, j))  # Сегмент + координати
    return segments, h // segment_size, w // segment_size


# Функція для обчислення ентропії сегмента
def calculate_entropy(segment):
    histogram, _ = np.histogram(segment, bins=256, range=(0, 255))
    return entropy(histogram, base=2)


# Функція для класифікації сегментів за ентропією
def classify_segments(segments):
    low_entropy_segments = []
    medium_entropy_segments = []
    high_entropy_segments = []

    entropies = [calculate_entropy(seg[0]) for seg in segments]
    threshold_low = np.percentile(entropies, 33)
    threshold_high = np.percentile(entropies, 66)

    for (segment, i, j), ent in zip(segments, entropies):
        if ent <= threshold_low:
            low_entropy_segments.append((segment, i, j))
        elif ent <= threshold_high:
            medium_entropy_segments.append((segment, i, j))
        else:
            high_entropy_segments.append((segment, i, j))

    return low_entropy_segments, medium_entropy_segments, high_entropy_segments


# Функція для апроксимації сегментів по рядках
def approximate_rows(segment):
    x = np.arange(segment.shape[1]).reshape(-1, 1)  # Координати стовпців
    restored_segment = np.zeros_like(segment, dtype=float)
    mse_list = []
    for row in range(segment.shape[0]):
        y = segment[row, :]  # Значення яскравості рядка
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        restored_segment[row, :] = y_pred
        mse_list.append(mean_squared_error(y, y_pred))
    return restored_segment, mse_list


# Функція для апроксимації сегментів по стовпцях
def approximate_columns(segment):
    x = np.arange(segment.shape[0]).reshape(-1, 1)  # Координати рядків
    restored_segment = np.zeros_like(segment, dtype=float)
    mse_list = []
    for col in range(segment.shape[1]):
        y = segment[:, col]  # Значення яскравості стовпця
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        restored_segment[:, col] = y_pred
        mse_list.append(mean_squared_error(y, y_pred))
    return restored_segment, mse_list


# Функція для повного відновлення зображення
def full_restore(image, segments, rows, cols, segment_size, method="rows"):
    restored_image = image.copy().astype(float)

    for segment, i, j in tqdm(segments, total=len(segments), desc=f"Restoring Image ({method})"):
        if method == "rows":
            restored_segment, _ = approximate_rows(segment)
        elif method == "columns":
            restored_segment, _ = approximate_columns(segment)
        else:
            raise ValueError("Invalid method. Use 'rows' or 'columns'.")

        restored_image[i:i + segment.shape[0], j:j + segment.shape[1]] = restored_segment

    return restored_image


# Функція для створення маски видалених зон
def create_removed_zones_mask(image_shape, segments):
    mask = np.ones(image_shape, dtype=np.uint8) * 255  # 255 для видимих областей
    for _, i, j in segments:
        mask[i:i + segment_size, j:j + segment_size] = 0  # 0 для видалених сегментів
    return mask


## Функція для обробки сегментів і виводу результатів
def process_segments(image, segments, rows, cols, segment_size, entropy_label):
    restored_image_rows = image.copy().astype(float)  # Початкове зображення для рядків
    restored_image_cols = image.copy().astype(float)  # Початкове зображення для стовпців
    mse_heatmap_rows = np.zeros((rows, cols))
    mse_heatmap_cols = np.zeros((rows, cols))
    total_mse_rows = []
    total_mse_cols = []

    for idx, (segment, i, j) in tqdm(enumerate(segments), total=len(segments),
                                     desc=f"Processing {entropy_label} Entropy Segments"):
        # Індекс сегмента в тепловій карті
        row_idx = i // segment_size
        col_idx = j // segment_size

        # Апроксимація по рядках
        restored_rows, mse_rows = approximate_rows(segment)
        restored_image_rows[i:i + segment.shape[0], j:j + segment.shape[1]] = restored_rows
        mse_heatmap_rows[row_idx, col_idx] = np.mean(mse_rows)
        total_mse_rows.extend(mse_rows)

        # Апроксимація по стовпцях
        restored_cols, mse_cols = approximate_columns(segment)
        restored_image_cols[i:i + segment.shape[0], j:j + segment.shape[1]] = restored_cols
        mse_heatmap_cols[row_idx, col_idx] = np.mean(mse_cols)
        total_mse_cols.extend(mse_cols)

    return restored_image_rows, restored_image_cols, total_mse_rows, total_mse_cols, mse_heatmap_rows, mse_heatmap_cols


# Функція для візуалізації регресії рядків
def visualize_row_regression(segment, restored_segment, entropy_label):
    x = np.arange(segment.shape[1])
    num_rows = segment.shape[0]
    mse = np.mean((segment - restored_segment) ** 2)
    rmse = np.sqrt(mse)  # Розрахунок СКВ

    # Кількість графіків (рядків + 2) і колонок
    total_plots = num_rows + 2
    cols = 2
    rows = (total_plots + cols - 1) // cols  # Округлення до більшого цілого

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, rows * 2.5))
    axes = axes.flatten()  # Перетворюємо у плоский список для зручності

    for row in range(num_rows):
        ax = axes[row]
        ax.plot(x, segment[row, :], label='Original Row')
        ax.plot(x, restored_segment[row, :], label='Restored Row', linestyle='--')
        ax.set_title(f'{entropy_label} Entropy: Row {row + 1}')
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Intensity')
        ax.legend()

    # Додаємо оригінальний сегмент
    ax_segment = axes[num_rows]
    ax_segment.imshow(segment, cmap='gray')
    ax_segment.set_title(f'{entropy_label} Entropy: Original Segment')
    ax_segment.axis('off')

    # Додаємо відновлений сегмент
    ax_restored = axes[num_rows + 1]
    ax_restored.imshow(restored_segment, cmap='gray')
    ax_restored.set_title(f'{entropy_label} Entropy: Restored Segment (RMSE = {rmse:.2f})')
    ax_restored.axis('off')

    # Ховаємо зайві subplot'и, якщо вони є
    for idx in range(num_rows + 2, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# Функція для візуалізації регресії стовпців
def visualize_column_regression(segment, restored_segment, entropy_label):
    x = np.arange(segment.shape[0])
    num_cols = segment.shape[1]
    mse = np.mean((segment - restored_segment) ** 2)
    rmse = np.sqrt(mse)  # Розрахунок СКВ

    # Кількість графіків (стовпців + 2) і колонок
    total_plots = num_cols + 2
    cols = 2
    rows = (total_plots + cols - 1) // cols  # Округлення до більшого цілого

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, rows * 2.5))
    axes = axes.flatten()  # Перетворюємо у плоский список для зручності

    for col in range(num_cols):
        ax = axes[col]
        ax.plot(x, segment[:, col], label='Original Column')
        ax.plot(x, restored_segment[:, col], label='Restored Column', linestyle='--')
        ax.set_title(f'{entropy_label} Entropy: Column {col + 1}')
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Intensity')
        ax.legend()

    # Оригінальний сегмент
    ax_original = axes[num_cols]
    ax_original.imshow(segment, cmap='gray')
    ax_original.set_title(f'{entropy_label} Entropy: Original Segment')
    ax_original.axis('off')

    # Відновлений сегмент
    ax_restored = axes[num_cols + 1]
    ax_restored.imshow(restored_segment, cmap='gray')
    ax_restored.set_title(f'{entropy_label} Entropy: Restored Segment (RMSE = {rmse:.2f})')
    ax_restored.axis('off')

    # Ховаємо зайві subplot'и, якщо вони є
    for idx in range(num_cols + 2, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


# Функція для візуалізації результатів з додаванням маски видалених зон
def visualize_results(original_image, restored_rows, restored_cols, mse_rows, mse_cols, removed_zones_mask, entropy_type):
    plt.figure(figsize=(24, 6))

    # Оригінальне зображення
    plt.subplot(1, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Оригінальне зображення")
    plt.axis('off')

    # Відновлення по рядках
    plt.subplot(1, 4, 2)
    plt.imshow(restored_rows, cmap='gray')
    plt.title(f"Відновлене по рядках ({entropy_type} ентропія)\nСереднє СКВ: {np.mean(mse_rows):.4f}")
    plt.axis('off')

    # Відновлення по стовпцях
    plt.subplot(1, 4, 3)
    plt.imshow(restored_cols, cmap='gray')
    plt.title(f"Відновлене по стовпцях ({entropy_type} ентропія)\nСереднє СКВ: {np.mean(mse_cols):.4f}")
    plt.axis('off')

    # Відображення маски видалених зон
    plt.subplot(1, 4, 4)
    plt.imshow(removed_zones_mask, cmap='gray')
    plt.title(f"Маска видалених зон ({entropy_type} ентропія)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Функція для візуалізації теплових мап
def visualize_heatmaps(heatmap_rows, heatmap_cols):
    plt.figure(figsize=(12, 6))

    # Теплова карта СКВ по рядках
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap_rows, cmap='hot', interpolation='nearest')
    plt.title("Теплова карта СКВ (рядки)")
    plt.colorbar()

    # Теплова карта СКВ по стовпцях
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_cols, cmap='hot', interpolation='nearest')
    plt.title("Теплова карта СКВ (стовпці)")
    plt.colorbar()

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()


# Функція для візуалізації гістограм
def visualize_histograms(mse_rows, mse_cols):
    plt.figure(figsize=(12, 6))

    # Гістограма розподілу СКВ (рядки)
    plt.subplot(1, 2, 1)
    plt.hist(mse_rows, bins=20, color='blue', alpha=0.7)
    plt.title("Гістограма СКВ (рядки)")
    plt.xlabel("СКВ")
    plt.ylabel("Частота")
    plt.grid(True)

    # Гістограма розподілу СКВ (стовпці)
    plt.subplot(1, 2, 2)
    plt.hist(mse_cols, bins=20, color='green', alpha=0.7)
    plt.title("Гістограма СКВ (стовпці)")
    plt.xlabel("СКВ")
    plt.ylabel("Частота")
    plt.grid(True)

    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    plt.show()


# Основна функція
def main(image_path, segment_size):
    # Зчитуємо зображення
    image = load_image(image_path)
    brightness_channel = extract_brightness_channel(image)

    # Поділяємо на сегменти
    segments, rows, cols = segment_image(brightness_channel, segment_size)

    # Класифікація сегментів за ентропією
    low_entropy_segments, medium_entropy_segments, high_entropy_segments = classify_segments(segments)

    # Обробка сегментів низької ентропії
    restored_rows_low, restored_cols_low, mse_rows_low, mse_cols_low, heatmap_rows_low, heatmap_cols_low = process_segments(
        brightness_channel, low_entropy_segments, rows, cols, segment_size, "Low"
    )

    # Обробка сегментів середньої ентропії
    restored_rows_medium, restored_cols_medium, mse_rows_medium, mse_cols_medium, heatmap_rows_medium, heatmap_cols_medium = process_segments(
        brightness_channel, medium_entropy_segments, rows, cols, segment_size, "Medium"
    )

    # Обробка сегментів високої ентропії
    restored_rows_high, restored_cols_high, mse_rows_high, mse_cols_high, heatmap_rows_high, heatmap_cols_high = process_segments(
        brightness_channel, high_entropy_segments, rows, cols, segment_size, "High"
    )

    # Візуалізація результатів регресії для сегментів з низькою ентропією
    for segment, i, j in low_entropy_segments:
        restored_rows, _ = approximate_rows(segment)
        visualize_row_regression(segment, restored_rows, "Low")
        restored_cols, _ = approximate_columns(segment)
        visualize_column_regression(segment, restored_cols, "Low")
        break  # Візуалізуємо лише один сегмент для прикладу

    # Візуалізація результатів регресії для сегментів із середньою ентропією
    for segment, i, j in medium_entropy_segments:
        restored_rows, _ = approximate_rows(segment)
        visualize_row_regression(segment, restored_rows, "Medium")
        restored_cols, _ = approximate_columns(segment)
        visualize_column_regression(segment, restored_cols, "Medium")
        break  # Візуалізуємо лише один сегмент для прикладу

    # Візуалізація результатів регресії для сегментів із високою ентропією
    for segment, i, j in high_entropy_segments:
        restored_rows, _ = approximate_rows(segment)
        visualize_row_regression(segment, restored_rows, "High")
        restored_cols, _ = approximate_columns(segment)
        visualize_column_regression(segment, restored_cols, "High")
        break  # Візуалізуємо лише один сегмент для прикладу

    # Вивід загальної статистики
    print("\nСтатистика СКВ для сегментів низької ентропії:")
    print(
        f"  Рядки: Середнє={np.mean(mse_rows_low):.4f}, Мінімум={np.min(mse_rows_low):.4f}, Максимум={np.max(mse_rows_low):.4f}")
    print(
        f"  Стовпці: Середнє={np.mean(mse_cols_low):.4f}, Мінімум={np.min(mse_cols_low):.4f}, Максимум={np.max(mse_cols_low):.4f}")

    print("\nСтатистика СКВ для сегментів середньої ентропії:")
    print(
        f"  Рядки: Середнє={np.mean(mse_rows_medium):.4f}, Мінімум={np.min(mse_rows_medium):.4f}, Максимум={np.max(mse_rows_medium):.4f}")
    print(
        f"  Стовпці: Середнє={np.mean(mse_cols_medium):.4f}, Мінімум={np.min(mse_cols_medium):.4f}, Максимум={np.max(mse_cols_medium):.4f}")

    print("\nСтатистика СКВ для сегментів високої ентропії:")
    print(
        f"  Рядки: Середнє={np.mean(mse_rows_high):.4f}, Мінімум={np.min(mse_rows_high):.4f}, Максимум={np.max(mse_rows_high):.4f}")
    print(
        f"  Стовпці: Середнє={np.mean(mse_cols_high):.4f}, Мінімум={np.min(mse_cols_high):.4f}, Максимум={np.max(mse_cols_high):.4f}")

    # Маски видалених зон для різних ентропій
    low_entropy_mask = create_removed_zones_mask(brightness_channel.shape, low_entropy_segments)
    medium_entropy_mask = create_removed_zones_mask(brightness_channel.shape, medium_entropy_segments)
    high_entropy_mask = create_removed_zones_mask(brightness_channel.shape, high_entropy_segments)

    # Візуалізація результатів із масками
    visualize_results(brightness_channel, restored_rows_low, restored_cols_low, mse_rows_low, mse_cols_low,
                      low_entropy_mask, "Низька")
    visualize_results(brightness_channel, restored_rows_medium, restored_cols_medium, mse_rows_medium, mse_cols_medium,
                      medium_entropy_mask, "Середня")
    visualize_results(brightness_channel, restored_rows_high, restored_cols_high, mse_rows_high, mse_cols_high,
                      high_entropy_mask, "Висока")

    # Повне відновлення за всіма сегментами
    fully_restored_rows = full_restore(brightness_channel, segments, rows, cols, segment_size, method="rows")
    fully_restored_columns = full_restore(brightness_channel, segments, rows, cols, segment_size, method="columns")

    # Візуалізація результатів
    plt.figure(figsize=(18, 8))

    # Оригінальне зображення
    plt.subplot(1, 3, 1)
    plt.imshow(brightness_channel, cmap='gray')
    plt.title("Оригінальне зображення")
    plt.axis('off')

    # Повне відновлення за рядками
    plt.subplot(1, 3, 2)
    plt.imshow(fully_restored_rows, cmap='gray')
    plt.title("Повне відновлення (рядки)")
    plt.axis('off')

    # Повне відновлення за стовпцями
    plt.subplot(1, 3, 3)
    plt.imshow(fully_restored_columns, cmap='gray')
    plt.title("Повне відновлення (стовпці)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # # Візуалізація теплових мап
    # visualize_heatmaps(heatmap_rows_low, heatmap_cols_low)
    # visualize_heatmaps(heatmap_rows_medium, heatmap_cols_medium)
    # visualize_heatmaps(heatmap_rows_high, heatmap_cols_high)
    #
    # # Візуалізація гістограм
    # visualize_histograms(mse_rows_low, mse_cols_low)
    # visualize_histograms(mse_rows_medium, mse_cols_medium)
    # visualize_histograms(mse_rows_high, mse_cols_high)


# Функція для створення теплової мапи ентропії
def create_entropy_heatmap(image_shape, segments, entropy_classes, segment_size):
    heatmap = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)  # Теплова мапа RGB

    # Визначаємо кольори для кожного рівня ентропії
    low_color = [0, 0, 255]       # Синій для низької ентропії
    medium_color = [255, 255, 0]  # Жовтий для середньої ентропії
    high_color = [255, 0, 0]      # Червоний для високої ентропії

    # Призначаємо кольори сегментам
    for (i, j), entropy_class in entropy_classes.items():
        if entropy_class == "low":
            color = low_color
        elif entropy_class == "medium":
            color = medium_color
        elif entropy_class == "high":
            color = high_color
        else:
            continue  # Пропускаємо невідомі класи

        heatmap[i:i + segment_size, j:j + segment_size, :] = color

    return heatmap


# Основна функція
def main_with_entropy_heatmap(image_path, segment_size):
    # Зчитуємо зображення
    image = load_image(image_path)
    brightness_channel = extract_brightness_channel(image)

    # Поділяємо на сегменти
    segments, rows, cols = segment_image(brightness_channel, segment_size)

    # Класифікація сегментів за ентропією
    low_entropy_segments, medium_entropy_segments, high_entropy_segments = classify_segments(segments)

    # Створюємо словник класифікації ентропії
    entropy_classes = {}
    for segment, i, j in low_entropy_segments:
        entropy_classes[(i, j)] = "low"
    for segment, i, j in medium_entropy_segments:
        entropy_classes[(i, j)] = "medium"
    for segment, i, j in high_entropy_segments:
        entropy_classes[(i, j)] = "high"

    # Створюємо теплову мапу ентропії
    entropy_heatmap = create_entropy_heatmap(brightness_channel.shape, segments, entropy_classes, segment_size)

    # Візуалізуємо теплову мапу ентропії
    plt.figure(figsize=(10, 10))
    plt.imshow(entropy_heatmap)
    plt.title("Теплова мапа ентропії")
    plt.axis('off')

    # Додаємо легенду
    import matplotlib.patches as mpatches
    low_patch = mpatches.Patch(color='blue', label='Низька ентропія')
    medium_patch = mpatches.Patch(color='yellow', label='Середня ентропія')
    high_patch = mpatches.Patch(color='red', label='Висока ентропія')
    plt.legend(handles=[low_patch, medium_patch, high_patch], loc='lower right', fontsize=12)

    plt.show()


# Запуск програми з тепловою мапою ентропії
if __name__ == "__main__":
    image_path = "I17_01_1.bmp"  # Вкажіть шлях до зображення
    segment_size = int(input("Введіть розмір сегмента (позитивне ціле число): "))
    main_with_entropy_heatmap(image_path, segment_size)
    main(image_path, segment_size)
