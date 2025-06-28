# выполнить для загрузки всех необходимых библиотек

#!/bin/bash

echo "Устанавливаю зависимости из requirements.txt..."
pip install -r requirements.txt

echo "Загружаю модель spaCy для английского..."
python -m spacy download en_core_web_sm

echo "Загружаю модель spaCy для русского..."
python -m spacy download ru_core_news_sm

echo "Установка завершена."
