import os
import base64
import subprocess
import sys
import importlib
import platform
import glob
import random
import psutil
import socket
import time
import torch
import numpy as np
import cv2
import tensorflow as tf
from colorama import init, Fore
from tensorflow.keras import layers, Model, Input, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from typing import Dict, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass, field
import json

# Настройка логирования для отслеживания работы программы
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('utn.log')
    ]
)
logger = logging.getLogger(__name__)

# Конфигурация программы с настройками размеров изображений и требуемых модулей
@dataclass
class Config:
    # Размеры изображений для разных уровней качества
    IMAGE_SIZES: Dict[int, int] = field(default_factory=lambda: {
        1: 32,  # Низкое качество
        2: 64,  # Среднее качество
        3: 128  # Высокое качество
    })
    # Список необходимых модулей с их версиями
    REQUIRED_MODULES: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "colorama": None,
        "tensorflow": "2.10",
        "numpy": "1.26.4",
        "opencv-python": None,
        "psutil": None,
        "scikit-learn": None
    })

config = Config()

# Глобальные переменные для отслеживания состояния программы
error = False
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

def install_module(module: str, version: Optional[str] = None) -> None:
    """
    Устанавливает Python модуль с указанной версией.
    
    Args:
        module: Название модуля для установки
        version: Опциональная версия модуля
    """
    package = f"{module}=={version}" if version else module
    try:
        # Формируем команду для установки через pip
        pip_args = [
            sys.executable, "-m", "pip", "install", package,
            "--disable-pip-version-check", "--no-warn-script-location"
        ]
        
        # Добавляем специальный флаг для Linux систем
        if IS_LINUX:
            pip_args.append("--break-system-packages")
            
        # Выполняем установку модуля
        subprocess.run(
            pip_args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info(f"Установлен: {package}")
    except subprocess.CalledProcessError as e:
        global error
        error = True
        logger.error(f"Ошибка установки {package}: {str(e)}")

def check_internet() -> bool:
    """
    Проверяет наличие интернет-соединения через попытку подключения к DNS серверу Google.
    
    Returns:
        bool: True если есть соединение, False в противном случае
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def get_available_memory() -> Tuple[float, float, str]:
    """
    Получает информацию о доступной памяти GPU и CPU.
    
    Returns:
        Tuple[float, float, str]: (VRAM в ГБ, DRAM в ГБ, название GPU)
    """
    vram = 0.0
    gpu_name = "CPU"
    
    try:
        # Проверяем наличие CUDA GPU
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
        # Проверяем наличие ROCm (AMD GPU)
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            gpu_name = "AMD ROCm"
            try:
                import torch_hip
                vram = torch_hip.get_device_properties(0).total_memory / (1024**3)
            except:
                vram = 1.0  # Значение по умолчанию для AMD GPU
        elif tf.config.list_physical_devices('GPU'):
            gpu_name = "NVIDIA CUDA"
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception as e:
        logger.warning(f"Ошибка при определении GPU: {str(e)}")
    
    # Получаем информацию о доступной оперативной памяти
    dram = psutil.virtual_memory().available / (1024 ** 3)
    return vram, dram, gpu_name

def load_utkface_data(path: str, img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает и обрабатывает данные из датасета UTKFace.
    
    Args:
        path: Путь к директории с изображениями
        img_size: Размер изображений для ресайза
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, y_age, y_gender)
    """
    X, y_age, y_gender = [], [], []
    processed_count = 0
    error_count = 0
    
    try:
        # Отключаем предупреждения OpenCV для чистоты вывода
        cv2.setLogLevel(0)
        
        # Обрабатываем каждое изображение в директории
        for filename in glob.glob(os.path.join(path, "*.jpg")):
            try:
                basename = os.path.basename(filename)
                parts = basename.split('_')
                
                # Проверяем корректность имени файла
                if len(parts) < 3:
                    logger.warning(f"Пропущен файл {basename}: некорректный формат имени")
                    error_count += 1
                    continue
                
                # Извлекаем возраст и пол из имени файла
                try:
                    age = int(parts[0])
                    gender = int(parts[1])
                except ValueError:
                    logger.warning(f"Пропущен файл {basename}: некорректные значения возраста или пола")
                    error_count += 1
                    continue
                
                # Проверяем корректность значения пола
                if gender not in [0, 1]:
                    logger.warning(f"Пропущен файл {basename}: некорректное значение пола")
                    error_count += 1
                    continue
                
                # Загружаем и обрабатываем изображение
                img = cv2.imread(filename)
                if img is None:
                    logger.warning(f"Пропущен файл {basename}: не удалось загрузить изображение")
                    error_count += 1
                    continue
                
                # Преобразуем изображение в RGB и изменяем размер
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y_age.append(age / 100.0)  # Нормализуем возраст
                y_gender.append(gender)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Пропущен файл {basename}: {str(e)}")
                error_count += 1
                continue
        
        if processed_count == 0:
            raise ValueError("Не удалось обработать ни одного изображения")
        
        # Преобразуем списки в numpy массивы и нормализуем значения
        X = np.array(X, dtype="float32") / 255.0
        y_age = np.array(y_age, dtype="float32")
        y_gender = np.array(y_gender, dtype="float32")
        
        logger.info(f"Обработано изображений: {processed_count}")
        if error_count > 0:
            logger.warning(f"Пропущено изображений из-за ошибок: {error_count}")
        
        return X, y_age, y_gender
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {str(e)}")
        raise

def predict_image(model_path: str, image_path: str) -> None:
    """
    Предсказывает возраст и пол на изображении с использованием обученной модели.
    
    Args:
        model_path: Путь к сохраненной модели
        image_path: Путь к изображению для предсказания
    """
    try:
        # Запрашиваем у пользователя качество изображения
        img_size_ch = int(input("Выбери качество фото в проверке (1 - 32px, 2 - 64px, 3 - 128px): "))
        img_size = config.IMAGE_SIZES.get(img_size_ch)
        if not img_size:
            raise ValueError("Неверный выбор размера изображения")
        
        # Загружаем модель и изображение
        model = models.load_model(model_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Невозможно загрузить изображение")
        
        # Подготавливаем изображение для предсказания
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Получаем предсказания
        gender_pred, age_pred = model.predict(img)
        gender = "Мужчина" if gender_pred[0][0] < 0.5 else "Женщина"
        age = max(0, int(age_pred[0][0] * 100))
        
        print(f"Предсказанный пол: {gender}")
        print(f"Предсказанный возраст: {age}")
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        print("😢Ошибка при обработке изображения.")

def setup_environment() -> None:
    """
    Настраивает окружение для работы с GPU/CPU, устанавливает необходимые зависимости.
    """
    global error
    
    if __name__ == "__main__":
        # Устанавливаем необходимые модули
        for module, version in config.REQUIRED_MODULES.items():
            try:
                importlib.import_module(module)
            except ModuleNotFoundError:
                install_module(module, version)
        
        try:
            # Проверяем наличие AMD GPU
            is_amd = False
            try:
                import torch_hip
                is_amd = True
            except ImportError:
                pass

            # Устанавливаем PyTorch в зависимости от типа GPU
            if is_amd:
                torch_args = [
                    sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/rocm5.4.2",
                    "--disable-pip-version-check", "--no-warn-script-location"
                ]
            else:
                torch_args = [
                    sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "--disable-pip-version-check", "--no-warn-script-location"
                ]
            
            if IS_LINUX:
                torch_args.append("--break-system-packages")
                
            subprocess.run(torch_args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("PyTorch установлен")
        except subprocess.CalledProcessError as e:
            error = True
            logger.error(f"Ошибка установки PyTorch: {str(e)}")

def calculate_optimal_parameters(vram: float, dram: float) -> Dict[str, Union[int, float]]:
    """
    Рассчитывает оптимальные параметры для обучения на основе доступной памяти.
    
    Args:
        vram: Объем VRAM в ГБ
        dram: Объем DRAM в ГБ
    
    Returns:
        Dict с оптимальными параметрами для обучения
    """
    # Определяем максимальный размер изображения
    if vram >= 8 and dram >= 16:
        max_img_size = 128
    elif vram >= 4 and dram >= 8:
        max_img_size = 64
    else:
        max_img_size = 32
    
    # Рассчитываем оптимальный размер батча для GPU
    if vram >= 8:
        gpu_batch_size = min(128, int(vram * 8))
    elif vram >= 4:
        gpu_batch_size = min(64, int(vram * 6))
    else:
        gpu_batch_size = min(32, int(vram * 4))
    
    # Рассчитываем оптимальный размер батча для CPU
    if dram >= 32:
        cpu_batch_size = min(128, int(dram * 2))
    elif dram >= 16:
        cpu_batch_size = min(64, int(dram * 1.5))
    else:
        cpu_batch_size = min(32, int(dram))
    
    # Определяем оптимальный режим работы
    if vram >= 4 and dram >= 8:
        recommended_mode = "GPU" if vram >= 6 else "HYBRID"
    else:
        recommended_mode = "CPU"
    
    return {
        "max_img_size": max_img_size,
        "gpu_batch_size": gpu_batch_size,
        "cpu_batch_size": cpu_batch_size,
        "recommended_mode": recommended_mode
    }

def configure_tensorflow(gpu_name: str, vram: float, dram: float) -> Tuple[tf.distribute.Strategy, int, int]:
    """
    Настраивает TensorFlow для работы с GPU/CPU с автоматической оптимизацией.
    
    Args:
        gpu_name: Название GPU
        vram: Объем VRAM в ГБ
        dram: Объем DRAM в ГБ
    
    Returns:
        Tuple с настройками TensorFlow
    """
    optimal_params = calculate_optimal_parameters(vram, dram)
    
    print(f"\nРекомендуемые параметры для вашей конфигурации:")
    print(f"- Максимальный размер изображения: {optimal_params['max_img_size']}px")
    print(f"- Рекомендуемый режим работы: {optimal_params['recommended_mode']}")
    print(f"- Размер батча для GPU: {optimal_params['gpu_batch_size']}")
    print(f"- Размер батча для CPU: {optimal_params['cpu_batch_size']}\n")
    
    if gpu_name != "CPU":
        # Запрашиваем у пользователя режим работы
        while True:
            use_mode = input(f"Выберите режим (1 - GPU, 2 - CPU, 3 - GPU + CPU) [рекомендуется {optimal_params['recommended_mode']}]: ").strip()
            if use_mode in ["1", "2", "3"]:
                break
            print("Ошибка: введите 1, 2 или 3.")
        
        # Настраиваем TensorFlow в зависимости от выбранного режима
        if use_mode == "1" and vram > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
            if "CUDA" in gpu_name or (IS_LINUX and "AMD" in gpu_name):
                tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
            logger.info(f"Используется GPU: {gpu_name} с {vram:.2f} ГБ VRAM")
            batch_size = optimal_params['gpu_batch_size']
        elif use_mode == "2":
            strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logger.info(f"Используется CPU с {dram:.2f} ГБ DRAM")
            batch_size = optimal_params['cpu_batch_size']
        else:
            strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/cpu:0"])
            logger.info("Используется комбинированный режим GPU + CPU")
            batch_size = min(optimal_params['gpu_batch_size'], optimal_params['cpu_batch_size'])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info(f"GPU не найден, используется CPU с {dram:.2f} ГБ DRAM")
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
        batch_size = optimal_params['cpu_batch_size']
    
    return strategy, batch_size, optimal_params['max_img_size']

def train_model(X_train, y_gender_train, y_age_train, X_test, y_gender_test, y_age_test, batch_size: int, img_size: int) -> None:
    """
    Обучает модель для определения возраста и пола на изображениях.
    
    Args:
        X_train: Обучающие изображения
        y_gender_train: Метки пола для обучения
        y_age_train: Метки возраста для обучения
        X_test: Тестовые изображения
        y_gender_test: Метки пола для тестирования
        y_age_test: Метки возраста для тестирования
        batch_size: Размер батча
        img_size: Размер изображений
    """
    try:
        print("Создаю модель...")
        input = Input(shape=(img_size, img_size, 3))
        
        # Создаем архитектуру модели в зависимости от размера изображения
        if img_size >= 128:
            x = layers.Conv2D(64, (3, 3), activation='relu')(input)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(256, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
        elif img_size >= 64:
            x = layers.Conv2D(32, (3, 3), activation='relu')(input)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)
        else:
            x = layers.Conv2D(32, (3, 3), activation='relu')(input)
            x = layers.MaxPooling2D((2, 2))(x)
        
        # Добавляем полносвязные слои
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        gender_output = layers.Dense(1, activation='sigmoid', name='gender')(x)
        age_output = layers.Dense(1, activation='linear', name='age')(x)

        # Создаем модель
        model = Model(inputs=input, outputs=[gender_output, age_output])

        # Компилируем модель
        model.compile(
            optimizer='adam',
            loss={'gender': 'binary_crossentropy', 'age': 'mse'},
            metrics={'gender': 'accuracy', 'age': 'mae'}
        )

        print("Обучаю модель...")
        # Обучаем модель
        model.fit(
            X_train, [y_gender_train, y_age_train],
            batch_size=batch_size,
            epochs=50,
            validation_data=(X_test, [y_gender_test, y_age_test]),
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )

        print("Сохранение модели...")
        model.save("utkface_model.keras")
        print("🎉 Модель успешно сохранена!")
        
    except tf.errors.ResourceExhaustedError:
        logger.error("Недостаточно памяти. Попробуйте уменьшить размер изображений или использовать меньший размер батча.")
        raise ValueError("Недостаточно памяти для обучения модели. Попробуйте выбрать меньший размер изображений.")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}")
        raise

if __name__ == "__main__":
    # Основной блок программы
    setup_environment()
    
    if error == False:
        logger.info("Все модули готовы к использованию!")
    else:
        logger.error("Ошибка модулей!")
    
    # Настраиваем очистку экрана в зависимости от ОС
    if IS_WINDOWS:
        clear = lambda: os.system('cls')
    if IS_LINUX:
        clear = lambda: os.system('clear')
    
    clear()
    
    # Настраиваем уровень логирования
    tf.get_logger().setLevel('WARNING')
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    cv2.setLogLevel(0)
    
    # Получаем информацию о системе
    print(f"Обнаружена OC: {platform.system()}")
    vram, dram, gpu_name = get_available_memory()
    strategy, batch_size, max_img_size = configure_tensorflow(gpu_name, vram, dram)
    
    # Запрашиваем у пользователя действие
    while True:
        choice = input("Выберите действие (1 - Обучение модели, 2 - Тестирование модели): ").strip()
        if choice in ["1", "2"]:
            break
        print("Ошибка: неверный ввод. Выберите 1 или 2.")
    
    clear()
    
    if choice == "1":
        # Показываем доступные размеры изображений
        available_sizes = {k: v for k, v in config.IMAGE_SIZES.items() if v <= max_img_size}
        print(f"Доступные размеры изображений для вашей конфигурации:")
        for size_id, size in available_sizes.items():
            print(f"{size_id} - {size}px")
        
        # Запрашиваем размер изображений
        while True:
            try:
                img_size_ch = int(input(f"Выбери качество фото в датасете (1-{len(available_sizes)}): "))
                img_size = available_sizes.get(img_size_ch)
                if img_size:
                    break
                print(f"Ошибка: выберите один из доступных размеров: {', '.join(map(str, available_sizes.keys()))}")
            except ValueError:
                print("Ошибка: введите число")

        print("Загрузка данных...")
        # Загружаем и обрабатываем данные
        X, y_age, y_gender = load_utkface_data("./UTKFace", img_size)
        if len(X) == 0:
            raise ValueError("Ошибка: не удалось загрузить изображения. Проверьте путь к датасету.")

        print("Уникальные значения y_gender:", np.unique(y_gender))

        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = train_test_split(
            X, y_age, y_gender, test_size=0.2, random_state=42
        )
        
        # Обучаем модель
        train_model(X_train, y_gender_train, y_age_train, X_test, y_gender_test, y_age_test, batch_size, img_size)

    elif choice == "2":
        # Запрашиваем путь к изображению для тестирования
        image_path = input("Введите путь к изображению: ").strip()
        model_path = "utkface_model.keras"
        predict_image(model_path, image_path)