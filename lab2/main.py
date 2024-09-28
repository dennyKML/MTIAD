import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Зчитування RGB-зображення
def load_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


# Функція для обчислення ентропії Шеннона
def calculate_shannon_entropy(channel):
    histogram, _ = np.histogram(channel, bins=256, range=(0, 256))
    probabilities = histogram / np.sum(histogram)
    non_zero_probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))
    return entropy


# Функція для обчислення міри Хартлі
def calculate_hartley_measure(channel):
    unique_values = np.unique(channel)
    hartley_measure = np.log2(len(unique_values))
    return hartley_measure


# Функція для обчислення Марковського процесу першого порядку
# def calculate_first_order_markov(channel):
#     flattened_channel = channel.flatten()
#     transitions = np.zeros((256, 256))
#
#     for i in range(len(flattened_channel) - 1):
#         current_pixel = flattened_channel[i]
#         next_pixel = flattened_channel[i + 1]
#         transitions[current_pixel, next_pixel] += 1
#
#     transition_probabilities = transitions / np.sum(transitions)
#     return transition_probabilities


# Функція для обчислення Марковського процесу першого порядку
def calculate_first_order_markov(channel):
    flattened_channel = channel.flatten()
    transitions = np.zeros((256, 256))

    # Підрахунок кількості переходів між пікселями
    for i in range(len(flattened_channel) - 1):
        current_pixel = flattened_channel[i]
        next_pixel = flattened_channel[i + 1]
        transitions[current_pixel, next_pixel] += 1

    # Нормалізація кожного рядка окремо
    row_sums = np.sum(transitions, axis=1, keepdims=True)

    # Щоб уникнути ділення на 0, замінюємо нульові суми на 1 (бо немає переходів для цих значень)
    row_sums[row_sums == 0] = 1

    transition_probabilities = transitions / row_sums
    return transition_probabilities


# Функція для сегментації зображення
def segment_image(image, segment_size):
    h, w, _ = image.shape
    segment_height = h // segment_size[0]
    segment_width = w // segment_size[1]
    segments = []

    for i in range(segment_size[0]):
        for j in range(segment_size[1]):
            segment = image[i * segment_height:(i + 1) * segment_height,
                      j * segment_width:(j + 1) * segment_width]
            segments.append(segment)

    return segments


# Функція для аналізу сегментів
def analyze_segments(image, segment_size):
    segments = segment_image(image, segment_size)
    entropies = {'R': [], 'G': [], 'B': []}
    hartleys = {'R': [], 'G': [], 'B': []}
    markovs = {'R': [], 'G': [], 'B': []}

    for segment in segments:
        r_channel, g_channel, b_channel = segment[:, :, 0], segment[:, :, 1], segment[:, :, 2]

        # Ентропія
        entropies['R'].append(calculate_shannon_entropy(r_channel))
        entropies['G'].append(calculate_shannon_entropy(g_channel))
        entropies['B'].append(calculate_shannon_entropy(b_channel))

        # Міра Хартлі
        hartleys['R'].append(calculate_hartley_measure(r_channel))
        hartleys['G'].append(calculate_hartley_measure(g_channel))
        hartleys['B'].append(calculate_hartley_measure(b_channel))

        # Марковський процес
        markovs['R'].append(np.sum(calculate_first_order_markov(r_channel)))
        markovs['G'].append(np.sum(calculate_first_order_markov(g_channel)))
        markovs['B'].append(np.sum(calculate_first_order_markov(b_channel)))

    return {
        'avg_entropy': {color: np.mean(entropies[color]) for color in ['R', 'G', 'B']},
        'avg_hartley': {color: np.mean(hartleys[color]) for color in ['R', 'G', 'B']},
        'avg_markov': {color: np.mean(markovs[color]) for color in ['R', 'G', 'B']},
        'entropies': entropies,
        'hartleys': hartleys,
        'markovs': markovs
    }


# Функція для порівняння результатів
def compare_results(image, segment_size):
    print("\nAnalyzing whole image...")

    r_channel, g_channel, b_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    whole_image_entropy = {
        'R': calculate_shannon_entropy(r_channel),
        'G': calculate_shannon_entropy(g_channel),
        'B': calculate_shannon_entropy(b_channel)
    }
    whole_image_hartley = {
        'R': calculate_hartley_measure(r_channel),
        'G': calculate_hartley_measure(g_channel),
        'B': calculate_hartley_measure(b_channel)
    }
    whole_image_markov = {
        'R': np.sum(calculate_first_order_markov(r_channel)),
        'G': np.sum(calculate_first_order_markov(g_channel)),
        'B': np.sum(calculate_first_order_markov(b_channel))
    }

    print("Analyzing segments...")
    segment_results = analyze_segments(image, segment_size)

    print("\nComparison:")
    for color in ['R', 'G', 'B']:
        print(f"{color} Channel:")
        print(f"  Whole image Shannon Entropy: {whole_image_entropy[color]}")
        print(f"  Average segment Shannon Entropy: {segment_results['avg_entropy'][color]}")
        print(f"  Whole image Hartley Measure: {whole_image_hartley[color]}")
        print(f"  Average segment Hartley Measure: {segment_results['avg_hartley'][color]}")
        print(f"  Whole image Markov Process: {whole_image_markov[color]}")
        print(f"  Average segment Markov Process: {segment_results['avg_markov'][color]}\n")

    visualize_results(segment_results, segment_size, whole_image_entropy, whole_image_hartley, whole_image_markov)


# Функція для візуалізації результатів
def visualize_results(segment_results, segment_size, total_entropy, total_hartley, total_markov):
    fig = plt.figure(figsize=(18, 15))

    channels = ['R', 'G', 'B']
    colormap = ['Reds', 'Greens', 'Blues']

    # Створюємо 3D графіки для кожного каналу
    for idx, color in enumerate(channels):
        X, Y = np.meshgrid(np.arange(segment_size[0]), np.arange(segment_size[1]))
        ax1 = fig.add_subplot(3, 3, idx * 3 + 1, projection='3d')
        Z_entropy = np.array(segment_results['entropies'][color]).reshape(segment_size)
        ax1.plot_surface(X, Y, Z_entropy, cmap=colormap[idx])
        ax1.set_title(f'{color} Entropy')

        ax2 = fig.add_subplot(3, 3, idx * 3 + 2, projection='3d')
        Z_hartley = np.array(segment_results['hartleys'][color]).reshape(segment_size)
        ax2.plot_surface(X, Y, Z_hartley, cmap=colormap[idx])
        ax2.set_title(f'{color} Hartley Measure')

        ax3 = fig.add_subplot(3, 3, idx * 3 + 3, projection='3d')
        Z_markov = np.array(segment_results['markovs'][color]).reshape(segment_size)
        ax3.plot_surface(X, Y, Z_markov, cmap=colormap[idx])
        ax3.set_title(f'{color} Markov Process')

    plt.suptitle(f'Information Measures for RGB Channels', fontsize=16)
    plt.tight_layout()
    plt.show()


# Основна програма
if __name__ == "__main__":
    image_path = 'I17_01_1.bmp'
    image_rgb = load_image(image_path)

    print("Analyzing whole image and comparing with segments...\n")

    try:
        segment_height = int(input("Enter the height of each segment in pixels (e.g., 8): "))
        segment_width = int(input("Enter the width of each segment in pixels (e.g., 8): "))
        segment_size = (segment_height, segment_width)
        compare_results(image_rgb, segment_size)
    except ValueError:
        print("Invalid input. Please enter valid integers for segment size.")
