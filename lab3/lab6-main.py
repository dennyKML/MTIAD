import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


# Функція для зчитування зображення у форматі RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image.astype(np.uint8)  # Перетворюємо на uint8


# Функція для сегментації зображення
def segment_image(image, segment_size):
    segments = []
    h, w, _ = image.shape
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            segment = image[i:i + segment_size, j:j + segment_size]
            if segment.shape[0] == segment_size and segment.shape[1] == segment_size:
                segments.append(segment)
    return segments, len(segments)  # Повертаємо сегменти і їх кількість


# Функція для розрахунку ентропії Шеннона
def calculate_entropy(segment):
    entropies = []
    for i in range(3):  # Канали R, G, B
        hist, _ = np.histogram(segment[:, :, i].ravel(), bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]  # Уникати нульових частот
        entropies.append(entropy(hist))
    return np.mean(entropies)  # Повертаємо середню ентропію по всіх каналах


# Функція для класифікації сегментів за рівнем ентропії
def classify_entropy_segments(image, segments, entropies, segment_size):
    classified_image = np.zeros_like(image)
    h, w, _ = image.shape
    idx = 0
    for i in range(0, h, segment_size):
        for j in range(0, w, segment_size):
            if 4 > entropies[idx] >= 3.6:  # Висока ентропія
                color = [255, 0, 0]  # Червоний
            elif 3.6 > entropies[idx] >= 3.2:  # Середня ентропія
                color = [0, 255, 0]  # Зелений
            elif entropies[idx] < 3.2:  # Низька ентропія
                color = [0, 0, 255]  # Синій
            classified_image[i:i + segment_size, j:j + segment_size] = color
            idx += 1

    # Показуємо оригінальне зображення та класифіковане зображення
    show_classification_result("Classified by Entropy Level", image, classified_image)


# Функція для сегментації за яскравістю YCrCb
def classify_brightness_segments(image, block_size=8, low_threshold=50, medium_threshold=150):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel = img_ycrcb[:, :, 0]
    low_segment = (y_channel < low_threshold)
    medium_segment = (y_channel >= low_threshold) & (y_channel < medium_threshold)
    high_segment = (y_channel >= medium_threshold)
    segments = {"low": low_segment, "medium": medium_segment, "high": high_segment}
    colors = {
        "low": [0, 0, 255],
        "medium": [0, 255, 0],
        "high": [255, 0, 0]
    }
    segmentation_image = np.zeros((*y_channel.shape, 3), dtype=np.uint8)
    h, w = y_channel.shape
    for seg_type, segment_mask in segments.items():
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block_mask = segment_mask[i:i + block_size, j:j + block_size]
                if np.any(block_mask):
                    segmentation_image[i:i + block_size, j:j + block_size] = colors[seg_type]

    # Показуємо оригінальне зображення та сегментоване за яскравістю
    show_brightness_result("Segmented Image by Brightness (Y Channel)", image, segmentation_image)


# Функція для відображення зображень
def show_classification_result(title, original_image, classified_image):
    plt.figure(figsize=(12, 6))

    # Відображення оригінального зображення
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.astype(np.uint8))
    plt.axis('off')
    plt.title("Original Image")

    # Відображення класифікованого зображення
    plt.subplot(1, 2, 2)
    plt.imshow(classified_image.astype(np.uint8))
    plt.axis('off')
    plt.title(title)

    # Додаємо легенду внизу полотна
    plt.figtext(0.5, 0.01, 'Red: High Entropy | Green: Medium Entropy | Blue: Low Entropy',
                ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()


def show_brightness_result(title, original_image, segmentation_image):
    plt.figure(figsize=(12, 6))

    # Відображення оригінального зображення
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.astype(np.uint8))
    plt.axis('off')
    plt.title("Original Image")

    # Відображення сегментованого зображення
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_image.astype(np.uint8))
    plt.axis('off')
    plt.title(title)

    # Додаємо легенду внизу полотна
    plt.figtext(0.5, 0.01, 'Blue: Low Brightness | Green: Medium Brightness | Red: High Brightness',
                ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()


# Функція для відображення таблиці порогових значень
def show_threshold_table():
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('tight')
    ax.axis('off')

    # Дані таблиці порогових значень
    table_data = [
        ["Класифікація", "Порогові Значення", "Кольори"],
        ["Ентропія", "High: 3.6 - 4.0, Medium: 3.2 - 3.6, Low: < 3.2", "Red | Green | Blue"],
        ["Яскравість (Y Channel)", "High: >= 150, Medium: 50 - 150, Low: < 50", "Red | Green | Blue"]
    ]

    # Додаємо таблицю візуально до зображення
    ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.25, 0.5, 0.25])
    plt.title("Threshold Table for Classification")
    plt.show()


# Основна функція для запуску аналізу зображення
def analyze_image_with_classifications(image_path, segment_size=8):
    image = load_image(image_path)

    # Показуємо таблицю порогових значень
    show_threshold_table()

    # Класифікація за ентропією
    segments, total_segments = segment_image(image, segment_size)
    print(f"Загальна кількість сегментів: {total_segments}")
    entropies = [calculate_entropy(segment) for segment in segments]
    classify_entropy_segments(image, segments, entropies, segment_size)

    # Класифікація за яскравістю YCrCb
    classify_brightness_segments(image, block_size=segment_size)


# Виклик основної функції
if __name__ == '__main__':
    image_path = 'I17_01_1.bmp'
    segment_size = int(input("Введіть розмір сегмента (позитивне ціле число): "))
    analyze_image_with_classifications(image_path, segment_size)