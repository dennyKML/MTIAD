import socket
import cv2
import numpy as np


# Налаштування сервера
def start_server(host='127.0.0.1', port=5000):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Сервер запущено на {host}:{port}")

    conn, addr = server_socket.accept()
    print(f"Підключено до: {addr}")

    # Отримання розміру зображення
    data = conn.recv(1024)
    image_size = int(data.decode('utf-8'))
    conn.send(b'OK')  # Відправляємо підтвердження

    # Отримання самого зображення
    received_image = b''
    while len(received_image) < image_size:
        packet = conn.recv(4096)
        if not packet:
            break
        received_image += packet

    # Перетворення отриманих даних в numpy масив і RGB зображення
    image_array = np.frombuffer(received_image, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Закриваємо з'єднання
    conn.close()
    server_socket.close()

    # Виведення зображення
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow('Received Image', image_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# Запуск сервера
if __name__ == '__main__':
    start_server()
