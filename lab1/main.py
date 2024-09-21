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
    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b
    return grayscale_image.astype(np.uint8)


# 3. Функція порогової обробки зображення
def threshold_processing(grayscale_image, threshold):
    thresholded_image = np.where(grayscale_image > threshold, 255, 0).astype(np.uint8)

    # Побудова гістограми
    histogram, bins = np.histogram(grayscale_image, bins=256, range=[0, 256])

    return thresholded_image, histogram


# 4. Сегментація окремо для кожної компоненти RGB та побудова гістограм
def segment_image(grayscale_image, segment_size):
    h, w = grayscale_image.shape
    segments = []
    histograms = []

    segment_height = h // segment_size[0]
    segment_width = w // segment_size[1]

    for i in range(segment_size[0]):
        for j in range(segment_size[1]):
            # Вирізаємо сегмент
            segment = grayscale_image[i * segment_height:(i + 1) * segment_height,
                      j * segment_width:(j + 1) * segment_width]
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

    # 4. Сегментація зображення окремо для кожного каналу
    try:
        segment_rows = int(input("Enter the number of rows for segmentation (e.g., 4): "))
        segment_cols = int(input("Enter the number of columns for segmentation (e.g., 4): "))
        segment_size = (segment_rows, segment_cols)
    except ValueError:
        print("Invalid input. Please enter valid integers for segmentation size.")
        return

    # Передаємо сіре зображення
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
        plt.xticks(np.arange(0, 256, 15), rotation=45)  # Крок на осі X - 25

        plt.xlabel('Intensity Value')
        plt.ylabel('Pixel Count')

        plt.show()


# Виклик основної програми
if __name__ == "__main__":
    image_path = 'I17_01_1.bmp'  # Вкажіть шлях до вашого зображення
    threshold_value = 117  # Вибране порогове значення
    main(image_path, threshold_value)
