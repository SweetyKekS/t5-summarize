# выполнить для загрузки всех необходимых библиотек

```bash
#!/bin/bash

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm