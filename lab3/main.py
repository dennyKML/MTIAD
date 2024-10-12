import cv2
import socket
import numpy as np
import time
import matplotlib.pyplot as plt


# Функція для зчитування зображення у форматі RGB
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


# Функція для відправки зображення на сервер
def send_image(image, client_socket):
    _, image_encoded = cv2.imencode('.bmp', image)
    image_bytes = image_encoded.tobytes()
    image_size = str(len(image_bytes)).encode('utf-8')
    client_socket.send(image_size)
    response = client_socket.recv(2)
    if response == b'OK':
        client_socket.sendall(image_bytes)


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


# Функція для розрахунку середнього показника для сегментів
def calculate_average_metric_for_segments(segments, metric_function):
    metrics = [metric_function(segment) for segment in segments]
    return np.mean(metrics, axis=0)


# Функція для візуалізації порівняння для кожного каналу
def plot_comparison(title, metric_full, metric_avg_segments, ylabel):
    channels = ['R', 'G', 'B']
    plt.figure()
    x = np.arange(len(channels))
    width = 0.3

    plt.bar(x - width/2, metric_full, width, label='Full Image')
    plt.bar(x + width/2, metric_avg_segments, width, label='Avg Segments')

    plt.xticks(x, channels)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# Функція для оцінки часу передачі зображення через протокол
def calculate_transfer_time_over_protocol(image, host='127.0.0.1', port=5000):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    start_time = time.time()
    send_image(image, client_socket)
    end_time = time.time()

    client_socket.close()
    return end_time - start_time


# Функція для накладання моделі помилок (випадковий шум)
def apply_noise(image, noise_level=100):
    noise = np.random.randint(0, noise_level, image.shape, dtype='uint8')
    noisy_image = image + noise
    return noisy_image


# Додавання Гауссового шуму до зображення
def add_gaussian_noise(image, mean=0, std=100):
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# Функція для підрахунку кількості пікселів із помилками
def count_pixel_errors(original_image, noisy_image):
    return np.sum(original_image != noisy_image)


# Функція для знаходження максимального значення помилки
def calculate_max_error(original_image, noisy_image):
    return np.max(np.abs(original_image - noisy_image))


# Функція для розрахунку коефіцієнта нормованої кореляції (по кожному каналу)
def calculate_correlation(original_image, noisy_image):
    correlations = []
    for i in range(3):  # Канали R, G, B
        mean_orig = np.mean(original_image[:, :, i])
        mean_noisy = np.mean(noisy_image[:, :, i])

        numerator = np.sum((original_image[:, :, i] - mean_orig) * (noisy_image[:, :, i] - mean_noisy))
        denominator = np.sqrt(np.sum((original_image[:, :, i] - mean_orig) ** 2) * np.sum((noisy_image[:, :, i] - mean_noisy) ** 2))

        correlation = numerator / denominator
        correlations.append(float(correlation))
    return correlations


# Функція для оцінки середньоквадратичного відхилення (RMSE по кожному каналу)
def calculate_rmse(original_image, noisy_image):
    rmses = []
    for i in range(3):  # Канали R, G, B
        mse = np.mean((original_image[:, :, i] - noisy_image[:, :, i]) ** 2)
        rmses.append(float(np.sqrt(mse)))
    return rmses


# Функція для розрахунку пікового відношення сигнал/шум (PSNR по кожному каналу)
def calculate_psnr(original_image, noisy_image, max_pixel_value=255):
    psnrs = []
    for i in range(3):  # Канали R, G, B
        mse = np.mean((original_image[:, :, i] - noisy_image[:, :, i]) ** 2)
        if mse == 0:
            psnrs.append(float('inf'))
        else:
            psnrs.append(float(10 * np.log10(max_pixel_value ** 2 / mse)))
    return psnrs


# Функція для відображення зображення
def show_image(title, image):
    plt.figure()
    plt.imshow(image.astype(np.uint8))
    plt.axis('off')  # Вимкнути осі
    plt.title(title)
    plt.show()


# Основна функція для запуску аналізу зображення
def analyze_image(image_path, segment_size=8):
    image = load_image(image_path)
    image = np.array(image, dtype=np.float64)

    # Відображення оригінального зображення
    show_image("Original Image", image)

    transfer_time = calculate_transfer_time_over_protocol(image)
    print(f"\nЧас передачі зображення через протокол: {transfer_time:.6f} секунд")

    # noisy_image = apply_noise(image)
    noisy_image = add_gaussian_noise(image)

    # Відображення шумного зображення
    show_image("Noisy Image", noisy_image)

    # Підрахунок кількості пікселів із помилками
    pixel_errors = count_pixel_errors(image, noisy_image)
    print(f"Кількість пікселів із помилками: {pixel_errors}")

    max_error = calculate_max_error(image, noisy_image)
    print(f"Максимальне значення помилки: {max_error}")

    # Розрахунок для всього зображення
    correlation = calculate_correlation(image, noisy_image)
    print(f"\nКоефіцієнт нормованої кореляції (R, G, B): {correlation}")

    rmse = calculate_rmse(image, noisy_image)
    print(f"Середньоквадратичне відхилення (RMSE) для каналів (R, G, B): {rmse}")

    psnr = calculate_psnr(image, noisy_image)
    print(f"Пікове відношення сигнал/шум (PSNR) для каналів (R, G, B): {psnr}")

    # Сегментація
    segments = segment_image(image, segment_size)
    noisy_segments = segment_image(noisy_image, segment_size)

    # Обчислення усереднених показників для сегментів
    avg_correlation = calculate_average_metric_for_segments(
        zip(segments, noisy_segments), lambda pair: calculate_correlation(pair[0], pair[1])
    )
    avg_rmse = calculate_average_metric_for_segments(
        zip(segments, noisy_segments), lambda pair: calculate_rmse(pair[0], pair[1])
    )
    avg_psnr = calculate_average_metric_for_segments(
        zip(segments, noisy_segments), lambda pair: calculate_psnr(pair[0], pair[1])
    )

    print(f"\nСередній коефіцієнт нормованої кореляції для сегментів (R, G, B): {avg_correlation}")
    print(f"Середнє середньоквадратичне відхилення (RMSE) для сегментів (R, G, B): {avg_rmse}")
    print(f"Середнє пікове відношення сигнал/шум (PSNR) для сегментів (R, G, B): {avg_psnr}")

    # Візуалізація порівнянь для кожного каналу
    plot_comparison('Correlation Comparison', correlation, avg_correlation, 'Correlation')
    plot_comparison('RMSE Comparison', rmse, avg_rmse, 'RMSE')
    plot_comparison('PSNR Comparison', psnr, avg_psnr, 'PSNR')


# Виклик основної функції
if __name__ == '__main__':
    image_path = 'I17_01_1.bmp'

    # Введення розміру сегмента з клавіатури
    segment_size = int(input("Введіть розмір сегмента (позитивне ціле число): "))

    analyze_image(image_path, segment_size)
