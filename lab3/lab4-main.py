import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# Функція для зчитування зображення у форматі RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


# Функція для сегментації зображення
def segment_image(image, segment_size):
    segments = []
    h, w, _ = image.shape
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            segment = image[i:i + segment_size, j:j + segment_size]
            if segment.shape[0] == segment_size and segment.shape[1] == segment_size:
                segments.append(segment)
    return segments


# Функція для розрахунку ентропії Шенона
def calculate_entropy(segment):
    entropies = []
    for i in range(3):  # Канали R, G, B
        hist, _ = np.histogram(segment[:, :, i].ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]  # Уникати нульових частот
        entropies.append(entropy(hist))
    return np.mean(entropies)  # Повертаємо середню ентропію по всіх каналах


# Функція для розрахунку середньоквадратичного відхилення (RMSE) від середнього значення для сегмента
def calculate_rmse(segment):
    rmses = []
    for i in range(3):  # Канали R, G, B
        mean_value = np.mean(segment[:, :, i])
        mse = np.mean((segment[:, :, i] - mean_value) ** 2)
        rmses.append(float(np.sqrt(mse)))
    return np.mean(rmses)  # Повертаємо середній RMSE по всіх каналах


# Функція для розрахунку коефіцієнта нормованої кореляції всередині сегмента між сусідніми пікселями
def calculate_correlation(segment):
    correlations = []
    for i in range(3):  # Канали R, G, B
        segment_channel = segment[:, :, i]
        mean_value = np.mean(segment_channel)

        # Розрахунок кореляції між сусідніми пікселями
        shifted_segment = np.roll(segment_channel, shift=1, axis=0)
        numerator = np.sum((segment_channel - mean_value) * (shifted_segment - mean_value))
        denominator = np.sqrt(np.sum((segment_channel - mean_value) ** 2) * np.sum((shifted_segment - mean_value) ** 2))

        if denominator != 0:
            correlation = numerator / denominator
        else:
            correlation = 0  # Якщо знаменник дорівнює нулю, вважаємо кореляцію нульовою
        correlations.append(correlation)
    return np.mean(correlations)  # Повертаємо середню кореляцію по всіх каналах


# Функція для візуалізації класифікації сегментів за метрикою з порогом
def visualize_segments(image, segments, metrics, threshold, title):
    classified_image = np.zeros_like(image)
    h, w, _ = image.shape
    idx = 0
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            if metrics[idx] >= threshold:
                classified_image[i:i + segment_size, j:j + segment_size] = image[i:i + segment_size, j:j + segment_size]
            idx += 1
    show_image(title, classified_image)


# Функція для відображення зображення
def show_image(title, image):
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')  # Вимкнути осі
    plt.title(title)
    plt.show()


# Функція для виведення графіків з метриками
def show_metric_heatmap(metric_data, title):
    plt.figure()
    plt.imshow(metric_data, cmap='jet')
    plt.colorbar()  # Додаємо кольорову шкалу
    plt.title(title)
    plt.show()


# Основна функція для запуску аналізу зображення з пороговими значеннями
def analyze_image_with_threshold(image_path, segment_size=8):
    image = load_image(image_path)
    image = np.array(image, dtype=np.float64)

    # Відображення оригінального зображення
    show_image("Original Image", image)

    # Сегментація зображення
    segments = segment_image(image, segment_size)

    # Класифікація за ентропією Шенона
    entropies = [calculate_entropy(segment) for segment in segments]
    entropy_map = np.array(entropies).reshape(image.shape[0] // segment_size, image.shape[1] // segment_size)
    show_metric_heatmap(entropy_map, 'Entropy Heatmap')
    visualize_segments(image, segments, entropies, threshold=3, title='Classified by Entropy')

    # Класифікація за RMSE від середнього значення
    rmses = [calculate_rmse(segment) for segment in segments]
    rmse_map = np.array(rmses).reshape(image.shape[0] // segment_size, image.shape[1] // segment_size)
    show_metric_heatmap(rmse_map, 'RMSE Heatmap')
    visualize_segments(image, segments, rmses, threshold=15.0, title='Classified by RMSE')

    # Класифікація за кореляцією між сусідніми пікселями
    correlations = [calculate_correlation(segment) for segment in segments]
    correlation_map = np.array(correlations).reshape(image.shape[0] // segment_size, image.shape[1] // segment_size)
    show_metric_heatmap(correlation_map, 'Correlation Heatmap')
    visualize_segments(image, segments, correlations, threshold=0.5, title='Classified by Correlation')


# Виклик основної функції
if __name__ == '__main__':
    image_path_1 = 'I17_01_1.bmp'
    image_path_2 = 'I14_01_1.bmp'

    # Введення розміру сегмента з клавіатури
    segment_size = int(input("Введіть розмір сегмента (позитивне ціле число): "))

    analyze_image_with_threshold(image_path_1, segment_size)
    analyze_image_with_threshold(image_path_2, segment_size)
