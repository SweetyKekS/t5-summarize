T5-small Summarizer с Gradio и spaCy (CPU)

Приложение для генерации краткого саммари из текста или `.txt` файла, использующее модель `t5-small`.

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

3. Текстовый файл the_bun.txt для проверки саммари текстовых файлов
