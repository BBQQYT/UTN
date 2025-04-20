# 🤖 Распознавание пола по лицу AI

## О проекте

Современные системы распознавания лиц часто требуют значительных ресурсов и могут быть недоступны для малого бизнеса или образовательных учреждений. Этот проект нацелен на создание простого, экономичного и эффективного решения для первичной классификации по полу, которое может работать на бюджетном оборудовании.

Проект реализует сверточную нейронную сеть (CNN) для анализа изображений лиц и предсказания пола.

**Ключевые цели:**

*   Разработать доступную и надежную AI-модель.
*   Обеспечить возможность работы на недорогих системах.
*   Создать проект с открытым кодом для гибкости и адаптации.

## Возможности

*   Автоматическая проверка и установка необходимых Python-пакетов.
*   Определение доступных ресурсов (CPU/GPU/VRAM/DRAM) и рекомендация оптимальных параметров.
*   Автоматическая загрузка и предобработка изображений из датасета UTKFace.
*   Построение и обучение свёрточной нейронной сети (CNN) для распознавания пола и возраста.
*   Использование механизма ранней остановки для предотвращения переобучения.
*   Сохранение обученной модели.
*   Интерактивный режим для обучения или тестирования модели на новых изображениях.

## Результаты

На текущем этапе проекта модель достигла точности классификации пола в **73%** на тестовой выборке датасета UTKFace.

Этот результат является базовой точкой и показывает принципиальную возможность решения задачи с помощью выбранного подхода. Анализ показал, что точность может быть значительно повышена за счет дальнейших улучшений архитектуры и подготовки данных.

## Технологии

*   **Язык:** Python
*   **Библиотеки:**
    *   TensorFlow / Keras
    *   OpenCV
    *   NumPy
    *   psutil
    *   colorama
    *   torch (используется для определения GPU)
    *   scikit-learn
*   **Датасет:** [UTKFace](https://susanqq.github.io/UTKFace/)

## Как начать

### Необходимые условия

*   Python 3.x
*   Подключение к Интернету (для автоматической установки зависимостей).
*   Датасет UTKFace. Скачайте изображения из датасета и поместите их в папку с названием `UTKFace` в корневой директории проекта.

### Установка и запуск

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/BBQQYT/UTN
    cd UTN
    ```

2.  **Поместите датасет:** Убедитесь, что папка `UTKFace` с изображениями находится в той же директории, что и файл `main.py`.

3.  **Запустите основной скрипт:**
    ```bash
    python main.py
    ```

Скрипт автоматически проверит и установит все необходимые Python-пакеты (TensorFlow, OpenCV и другие). Следуйте инструкциям в консоли, чтобы выбрать режим работы (обучение или тестирование).

## Использование

После запуска скрипта `python main.py` вам будет предложено выбрать действие:

*   **1 - Обучение модели:**
    *   Происходит загрузка и предобработка датасета UTKFace.
    *   На основе вашей конфигурации оборудования будут предложены оптимальные размеры изображений для обучения и размер батча.
    *   Модель будет обучена на тренировочной выборке.
    *   Обученная модель сохранится в файл `utkface_model.keras`.
*   **2 - Тестирование модели:**
    *   Загружается модель из файла `utkface_model.keras`.
    *   Вам будет предложено ввести путь к изображению для анализа.
    *   Модель выполнит предсказание пола и возраста для указанного изображения.

## Структура проекта
├── main.py # Основной скрипт
├── UTKFace/ # Папка для изображений датасета UTKFace (нужно скачать отдельно).
├── utkface_model.keras # Файл с сохраненной моделью (появляется после обучения).
├── utn.log # Лог-файл работы скрипта.
└── README.md # Этот файл.

## Направления дальнейших исследований и улучшений

*   Внедрение техник аугментации данных (случайные повороты, масштабирование, изменения яркости) для повышения устойчивости модели к вариациям во входных изображениях.
*   Эксперименты с более сложными и проверенными архитектурами CNN (например, ResNet, MobileNetV2) для улучшения извлечения признаков.
*   Оптимизация гиперпараметров обучения (learning rate, размер батча, patience ранней остановки).
*   Более детальный анализ ошибок и работа с "трудными" примерами из датасета.
*   Изучение возможности развертывания модели на маломощных устройствах (например, Raspberry Pi) для работы в реальном времени.
*   Повышение точности распознавания возраста.

## Автор

*   BBQQYT

## Руководитель

*   Жанна Леонидовна

## Лицензия

Этот проект распространяется под лицензией **GNU General Public License v3.0 (GPLv3)**. См. файл [LICENSE](LICENSE) для дополнительной информации.

---
---

# 🤖 Gender Recognition by Face AI

## About the Project

Modern facial recognition systems often require significant resources and can be inaccessible for small businesses or educational institutions. This project aims to create a simple, economical, and effective solution for primary gender classification that can run on budget hardware.

The project implements a Convolutional Neural Network (CNN) to analyze facial images and predict gender.

**Key Goals:**

*   Develop an accessible and reliable AI model.
*   Ensure compatibility with low-cost systems.
*   Create an open-source project for flexibility and adaptation.

## Features

*   Automatic checking and installation of required Python packages.
*   Detection of available resources (CPU/GPU/VRAM/DRAM) and recommendation of optimal parameters.
*   Automatic loading and preprocessing of images from the UTKFace dataset.
*   Building and training a Convolutional Neural Network (CNN) for gender and age recognition.
*   Using an early stopping mechanism to prevent overfitting.
*   Saving the trained model.
*   Interactive mode for training or testing the model on new images.

## Results

At the current stage of the project, the model achieved a gender classification accuracy of **73%** on the UTKFace dataset's test set.

This result serves as a baseline and demonstrates the feasibility of solving the problem using the chosen approach. Analysis indicates that accuracy can be significantly improved through further enhancements to the architecture and data preparation.

## Technologies

*   **Language:** Python
*   **Libraries:**
    *   TensorFlow / Keras
    *   OpenCV
    *   NumPy
    *   psutil
    *   colorama
    *   torch (used for GPU detection)
    *   scikit-learn
*   **Dataset:** [UTKFace](https://susanqq.github.io/UTKFace/)

## Getting Started

### Prerequisites

*   Python 3.x
*   Internet connection (for automatic dependency installation).
*   UTKFace dataset. Download the dataset images and place them in a folder named `UTKFace` in the root directory of the project.

### Installation and Running

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BBQQYT/UTN
    cd UTN
    ```

2.  **Place the dataset:** Ensure that the `UTKFace` folder containing the images is in the same directory as the `main.py` file.

3.  **Run the main script:**
    ```bash
    python main.py
    ```

The script will automatically check and install all necessary Python packages (TensorFlow, OpenCV, and others). Follow the instructions in the console to select the mode of operation (training or testing).

## Usage

After running the `python main.py` script, you will be prompted to select an action:

*   **1 - Train the model:**
    *   The UTKFace dataset is loaded and preprocessed.
    *   Based on your hardware configuration, optimal image sizes for training and batch size will be suggested.
    *   The model will be trained on the training set.
    *   The trained model will be saved to the `utkface_model.keras` file.
*   **2 - Test the model:**
    *   The model is loaded from the `utkface_model.keras` file.
    *   You will be prompted to enter the path to the image for analysis.
    *   The model will perform gender and age prediction for the specified image.

## Project Structure
├── main.py # Main script
├── UTKFace/ # Folder for UTKFace dataset images (needs to be downloaded separately).
├── utkface_model.keras # File with the saved model (appears after training).
├── utn.log # Log file of the script's operation.
└── README.md # This file.

## Future Work and Improvements

*   Implementing data augmentation techniques (random rotations, scaling, brightness changes) to improve model robustness to variations in input images.
*   Experimenting with more complex and proven CNN architectures (e.g., ResNet, MobileNetV2) for better feature extraction.
*   Optimizing training hyperparameters (learning rate, batch size, early stopping patience).
*   More detailed error analysis and working with "difficult" examples from the dataset.
*   Investigating the possibility of deploying the model on low-power devices (e.g., Raspberry Pi) for real-time operation.
*   Improving the accuracy of age recognition.

## Author

*   BBQQYT

## Supervisor

*   Zhanna L. Muromtseva

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the [LICENSE](LICENSE) file for more details.
