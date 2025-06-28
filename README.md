T5-small Summarizer с Gradio и spaCy (CPU)

Приложение для генерации краткого саммари из текста или `.txt` файла, использующее модель `t5-small`.

Так как предпочтение лёгким и бесплатным для запуска моделям, то выбрана очень легкая модель T5-small с 60М параметров. 
Запускается локально и на слабых компах(тестирование на RV508) на CPU.
Модель плохо обучена на русском языке, поэтому выдает абракадабру. Так как в ТЗ не указан с какими языками работает LLM, то для примера выбрана 
эта модель. Если входные данные используются на русском языке, лучше использовать другую модель(например, facebook/bart-large-cnn). 
Для инкапсуляции кода генерации и большей гибкости для саммари создадим класс.
Для работы с большими текстами используется механизм итеративной генерации + финального сжатия,
самый надёжный вариант для длинных и сложных текстов, гибко и стабильно.

Установка:

1. Клонируй репозиторий, выполнив команду в терминале:


git clone https://github.com/SweetyKekS/t5-summarize.git

cd t5-summarize

2. Для установки библиотек выполни следующие команды

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm

Для Linux, MacOS выполни bash:
chmod +x setup.sh
./setup.sh

3. Запускаем файл app.py  и следуем инструкции  gradio

4. Текстовый файл the_bun.txt для проверки саммари текстовых файлов

5. Примеры саммари:
    а) Английский ввод текста: 

    'Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks without explicit instructions.[1] Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance.[2]

    ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine.[3][4] The application of ML to business problems is known as predictive analytics.

    Statistics and mathematical optimisation (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning.[6][7]

    From a theoretical viewpoint, probably approximately correct learning provides a framework for describing machine learning.
    '

    Результат:

    '
    advances in deep learning have allowed neural networks to surpass many previous machine learning approaches in performance. despite advances gaining popularity in the fields of deep-learning and deepening the field'macroeconomics' ML is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalise to unseen data.
    '

    б) файл the_bun.txt

    Результат:
    '
    the bun sang: "little bun, I have become old now and hard of hearing" the old man said he was scraped from bin and swept the flour bin. the fox said she'd like to hear it again. "come sit on my tongue, sing it"
    '

    в) Русский текст:
    '
    Машинное обучение (англ. machine learning, ML) — класс методов искусственного интеллекта, характерной чертой которых является не прямое решение задачи, а обучение за счёт применения решений множества сходных задач. Для построения таких методов используются средства математической статистики, численных методов, математического анализа, методов оптимизации, теории вероятностей, теории графов, различные техники работы с данными в цифровой форме.

    Различают два типа обучения:

    Обучение по прецедентам, или индуктивное обучение, основано на выявлении эмпирических закономерностей в данных.
    Дедуктивное обучение предполагает формализацию знаний экспертов и их перенос в компьютер в виде базы знаний.
    Дедуктивное обучение принято относить к области экспертных систем, поэтому термины машинное обучение и обучение по прецедентам можно считать синонимами.

    Многие методы индуктивного обучения разрабатывались как альтернатива классическим статистическим подходам. Многие методы тесно связаны с извлечением информации (англ. information extraction, information retrieval), интеллектуальным анализом данных (data mining).
    '

    Результат:
    '
    методов искусственноо интеллекта. ксертов - аналиом данн (data mining) – моно ситат сван в математиеско, и теории веротносте.
    '