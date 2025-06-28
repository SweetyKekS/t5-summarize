import gradio as gr
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
import spacy


'''
Для разбиения больших текстов на предложения нашего текста используем spacy, ипользуя токенизатор en_core_web_sm. 
Так как предпочтение лёгким и бесплатным для запуска моделям, то выбрана очень легкая модель T5-small с 60М параметров. 
Запускается локально и на слабых компах(тестирование на RV508) на CPU.
Модель плохо обучена на русском языке, поэтому выдает абракадабру. Так как в ТЗ не указан с какими языками работает LLM, то для примера выбрана 
эта модель. Если входные данные используются на русском языке, лучше использовать другую модель(например, facebook/bart-large-cnn). 
Для инкапсуляции кода генерации и большей гибкости для саммари создадим класс.
Для работы с большими текстами используется механизм итеративной генерации + финального сжатия,
самый надёжный вариант для длинных и сложных текстов, гибко и стабильно.
'''

class T5Summarizer:
    def __init__(self, model_name='google-t5/t5-small', spacy_tokenizer='en_core_web_sm', device='cpu'):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.nlp = spacy.load(spacy_tokenizer)  # 'en_core_web_sm' для английского или 'ru_core_news_sm' для русского

    def summarize(self, text, max_length=150, min_length=50):
        # для T5 нужно обязательно указать задачу(summarize), также удаляем ненужные пробелы и заменяем пробелами перенос на другую строку
        input_text = 'summarize: ' + text.strip().replace('\n', ' ') 
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors='pt', #  возвращает pytorch tensor
            max_length=512, # максимальная длина последовательности
            truncation=True, # удаляет токены, если длина превышает max_length
            padding=True # дополняет последовательность до максимальной в пакете
        ).to(self.device)

        # для контроля генерации используем следующие параметры  класса GenerationConfig
        summary_ids = self.model.generate(
            inputs,
            num_beams=2, # на каждом шаге рассматриваем два лучших варианта(лучше использовать для суммаризации)
            length_penalty=1.5, # поощряем короткие ответы(лучше использовать для суммаризации)
            max_length=max_length, # максимальная длина выходной последовательности (в токенах)
            min_length=min_length, # минимальная длина выходной последовательности (в токенах)
            no_repeat_ngram_size=2, # ограничение повторения фраз из двух слов подряд
            early_stopping=True # генерация останавливается, как только появляется num_beams кандидатов завершается токеном <eos>
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    '''
    Разбиваем текст на предложения с помощью spacy,  создаем список чанков для будещей генерации, в каждом чанке не более 450 токенов,
    помним, что в Т5 на входе не более 512 токенов. 
    '''

    def text_to_chunks(self, text, max_tokens=450):
        spacy_text = self.nlp(text)
        sentences = [sent.text.strip() for sent in spacy_text.sents]
        chunks = []
        current_chunk = ''
        for sentence in sentences: # проходимся по каждому предложению
            temp = current_chunk + ' ' + sentence # создаем переменную для хранения чанка, не превышающего max_tokens
            if len(self.tokenizer.encode(temp)) > max_tokens: # если превышает, останавливаем добавление токенов
                if current_chunk:
                    chunks.append(current_chunk.strip()) # добавляем чанку в список чанков
                current_chunk = sentence #  продолжаем с последнего предложения
            else:
                current_chunk = temp
        if current_chunk:
            chunks.append(current_chunk.strip()) # добавляем последний накопившийся чанк
        return chunks
    '''
    Создание финальной саммари. Разделяем текст на чанки, делаем саммари каждой чанки и сохраняем в списке,
    объединяем чанки после саммари в текст и снова производим финальную саммари.
    '''
    def summarize_long_text(self, text):
        chunks = self.text_to_chunks(text)
        partial_summaries = [self.summarize(chunk) for chunk in chunks]
        final_text = ' '.join(partial_summaries)
        final_summary = self.summarize(final_text)
        return final_summary

    '''
    Метод для чтения загруженного файла .txt и его саммари
    '''

    def summarize_file(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.summarize_long_text(text)
    
summarize = T5Summarizer() # создаем экземпляр

'''
UI сделаем максимально простым, для низкоуровневой настройки используем класс Blocks,
в блоке будет две вкладки: одна для вставки текста и его саммари, другая для выбора файла .txt и его саммари.

'''
with gr.Blocks() as demo:
    gr.Markdown('T5-Small(CPU)')
    
    with gr.Tab('Ввести текст'):
        input_text = gr.Textbox(lines=10, label='Вставьте текст')
        btn_1 = gr.Button('Саммари')
        output_text_1 = gr.Textbox(label='Результат')

        btn_1.click(fn=summarize.summarize_long_text, inputs=input_text, outputs=output_text_1)

    with gr.Tab('Загрузить .txt файл'):
        file_input = gr.File(label='Выберите .txt файл')
        btn_2 = gr.Button('Саммари')
        output_text_2 = gr.Textbox(label='Результат')

        btn_2.click(fn=summarize.summarize_file, inputs=file_input, outputs=output_text_2)


demo.launch()
