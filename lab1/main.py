import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування RGB-зображення
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Перетворюємо на RGB
    return rgb_image


# 2. Функція для перетворення зображення на градації сірого
def convert_to_grayscale(rgb_image):
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]
    grayscale_image = 0.2126 * r + 0.7142 * g + 0.0722 * b
    return grayscale_image.astype(np.uint8)


# 3. Функція порогової обробки зображення
def threshold_processing(grayscale_image, threshold):
    thresholded_image = np.where(grayscale_image > threshold, 255, 0).astype(np.uint8)

    # Побудова гістограми
    histogram, bins = np.histogram(grayscale_image, bins=256, range=[0, 256])

    return thresholded_image, histogram


# 4. Сегментація та побудова гістограм
def segment_image(grayscale_image, segment_pixel_size):
    h, w = grayscale_image.shape
    segment_height, segment_width = segment_pixel_size
    segments = []
    histograms = []

    # Проходимо по зображенню і вирізаємо сегменти заданого розміру
    for i in range(0, h, segment_height):
        for j in range(0, w, segment_width):
            # Вирізаємо сегмент, який має розмір segment_pixel_size (наприклад, 8x8 пікселів)
            segment = grayscale_image[i:i + segment_height, j:j + segment_width]
            segments.append(segment)

            # Створюємо гістограму для кожного сегмента
            histogram, _ = np.histogram(segment, bins=256, range=[0, 256])
            histograms.append(histogram)

    return segments, histograms


# Основна програма
def main(image_path, threshold_value):
    # 1. Зчитуємо зображення
    image_rgb = load_image(image_path)

    # Відображаємо RGB-зображення
    plt.imshow(image_rgb)
    plt.title('RGB Image')
    plt.axis('off')
    plt.show()

    # 2. Перетворюємо на градації сірого
    grayscale_image = convert_to_grayscale(image_rgb)

    # Відображаємо чорно-біле зображення
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image (Manual)')
    plt.axis('off')
    plt.show()

    # 3. Порогова обробка
    thresholded_image, histogram = threshold_processing(grayscale_image, threshold_value)

    # Відображаємо порогове зображення
    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f'Thresholded Image (Threshold = {threshold_value})')
    plt.axis('off')
    plt.show()

    # Відображаємо гістограму
    plt.plot(histogram)
    plt.title('Histogram')
    plt.show()

    # 4. Сегментація зображення з фіксованим розміром сегментів
    try:
        segment_height = int(input("Enter the height of each segment in pixels (e.g., 8): "))
        segment_width = int(input("Enter the width of each segment in pixels (e.g., 8): "))
        segment_size = (segment_height, segment_width)
    except ValueError:
        print("Invalid input. Please enter valid integers for segment size.")
        return

    # Передаємо сіре зображення і розмір сегмента
    segments, histograms = segment_image(grayscale_image, segment_size)

    # Відображаємо сегменти і гістограми для градацій сірого зображення
    for i, (segment, histogram) in enumerate(zip(segments, histograms)):
        plt.figure(figsize=(12, 6))

        # Відображаємо сегмент
        plt.subplot(1, 2, 1)
        plt.imshow(segment, cmap='gray')
        plt.title(f'Grayscale Segment {i + 1}')
        plt.axis('off')

        # Відображаємо гістограму сегмента
        plt.subplot(1, 2, 2)
        plt.plot(histogram)
        plt.title(f'Histogram for Segment {i + 1}')

        # Налаштовуємо вісь
        plt.xticks(np.arange(0, 256, 15), rotation=45)  # Крок на осі X - 15

        plt.xlabel('Intensity Value')
        plt.ylabel('Pixel Count')

        plt.show()
        # time.sleep(1)


# Виклик основної програми
if __name__ == "__main__":
    image_path = 'I17_01_1.bmp'  # Вкажіть шлях до вашого зображення
    threshold_value = 117  # Вибране порогове значення
    main(image_path, threshold_value)
