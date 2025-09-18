# Задание 1

### Сравнение LLM-моделей

| Критерий                           | Локальные Hugging Face                                                      | OpenAI                                                            | YandexGPT                                                         |
| ---------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| Качество ответов                   | Качество варьируется в зависимости от модели, обычно ниже облачных аналогов | Высокое качество, лучшая креативность и понимание контекста       | Хорошее качество, оптимизирован под русский язык                  |
| Скорость работы                    | Зависит от аппаратного обеспечения                                          | Высокая скорость, низкая задержка                                 | Высокая скорость, низкая задержка                                 |
| Стоимость владения и использования | Высокие первоначальные затраты на hardware, но нет recurring costs          | Оплата по использованию (per token), нет затрат на инфраструктуру | Оплата по использованию (per token), нет затрат на инфраструктуру |
| Удобство и простота развёртывания  | Сложное: требуется настройка инфраструктуры, оптимизация и обслуживание     | Максимально простое, API готово к использованию                   | Максимально простое, API готово к использованию                   |

### Сравнение модели эмбеддингов

| Критерий                           | Локальные Sentence-Transformers                                    | Облачные OpenAI Embeddings          |
| ---------------------------------- | ------------------------------------------------------------------ | ----------------------------------- |
| Скорость создания индекса          | Зависит от локального аппаратного обеспечения                      | Высокая, но зависит от лимитов API  |
| Качество поиска                    | Хорошее качество, особенно у специализированных моделей            | Стабильно высокое качество          |
| Стоимость владения и использования | Единовременные затраты на аппаратное обеспечение, далее затрат нет | Оплата по использованию (per token) |

### Сравнение векторных баз

| Критерий                                 | FAISS                                                     | ChromaDB                                                   |
| ---------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| Скорость поиска и индексации             | Очень высокая скорость, оптимизирована для поиска         | Хорошая скорость, но обычно медленнее FAISS                |
| Сложность внедрения и поддержки          | Средняя: требует больше технических знаний для интеграции | Простая, есть встроенные функции управления                |
| Удобство в работе                        | Низкоуровневая, требует ручной реализации многих функций  | User-friendly: REST API, встроенное управление коллекциями |
| Стоимость владения (учёт инфраструктуры) | Практически нулевая                                       | Практически нулевая                                        |

### Рекомендованная конфигурация сервера для разворачивания RAG-бота

**Минимальная конфигурация (бюджетный вариант), подходит для MVP**

- CPU: 8-12 ядер (Intel Xeon или AMD EPYC)
- RAM: 32 GB DDR4
- GPU: NVIDIA RTX 4090
- Storage: 512 GB NVMe SSD

**Оптимальная конфигурация, для production:**

- CPU: 16-24 ядра (Intel Xeon или AMD EPYC)
- RAM: 64 GB DDR4
- GPU: 2× NVIDIA RTX 4090 (24 GB VRAM) или 1× A5000 (24 GB)
- Storage: 1 TB NVMe SSD

### Варианты архитектуры

**Локальный**

- LLM: Mistral 7B/Mixtral 8x7B

- Эмбеддинги: all-MiniLM-L6-v2

- Векторная БД: FAISS

- Плюсы: полный контроль данных, низкая стоимость владения

- Минусы: требует технической экспертизы, затраты на аппаратное обеспечение и обслуживание

**Гибридный (локальная LLM + облачные эмбеддинги)**

- LLM: Llama 2 70B или Mixtral 8x7B (локально)
- Эмбеддинги: OpenAI/Sentence-Transformers
- Векторная БД: ChromaDB/FAISS
- Плюсы: баланс стоимости и качества
- Минусы: сложность настройки

**Yandex Cloud Solution**

- LLM: YandexGPT
- Эмбеддинги: Yandex Embeddings
- Векторная БД: YDB
- Плюсы: простота, оптимизация под русский язык
- Минусы: vendor lock-in, большие затраты при масштабировании, нарушение конфиденциальности данных

Я бы рекомендовал остановиться на локальной архитектуре. И вот почему:

- Конфиденциальность данных. Данные не покидают сеть компании, следовательно, исключена возможность утечки чувствительной информации.

- Стоимость решения. Да, первоначальные затраты будут ощутимыми - необходимо приобрести аппаратное обеспечение для разворачивания бота. Но на долгой дистанции эти затраты всё равно будут ниже, чем оплата облачных сервисов. Особенно, если мы планируем расширение базы знаний.

- Масштабируемость. В случае необходимости мы можем выполнить масштабирование в соответствии с нашими потребностями.

# Задание 2

- Сырые данные лежат в каталоге `raw_data`. Скрипт для скачивания и очистки текста - [download_pages.py](./download_pages.py).

- Для базы знаний я взял часть вики, посвящённой циклу [Вечный Воитель](https://ru.wikipedia.org/wiki/%D0%92%D0%B5%D1%87%D0%BD%D1%8B%D0%B9_%D0%92%D0%BE%D0%B8%D1%82%D0%B5%D0%BB%D1%8C) [Майкла Муркока](https://ru.wikipedia.org/wiki/%D0%9C%D1%83%D1%80%D0%BA%D0%BE%D0%BA,_%D0%9C%D0%B0%D0%B9%D0%BA%D0%BB). 

- Словарь замен находится в файле [json/terms_map.json](./json/terms_map.json). Замены выбирались по следующему принципу: замены имён собственных получены посредством случайной генерации с использованием нескольких генераторов. Для понятий и названий предметов замены были прдуманы самостоятельно так, чтобы максимально сохранить связность и логику текста.

- Скрипт, производящий замены - [replace_terms.py](./replace_terms.py).

- Финальная база знаний, подготовленная к созданию векторного индекса, находится в каталоге `knowledge_base`.

# Задание 3

- Для построения индекса использовалась модель [sentence-transformers/all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

- База знаний - текстовые файлы из каталога `knowledge_base`.

- В построенном индексе 229 чанков. Размер чанка: 400, перекрытие: 50.

- Построение индекса заняло 6.223367 секунд.

- Скрипт для построения индекса - [build_index.py](./build_index.py).

- Примеры запросов к индексу можно посмотреть в скрипте [query_index.py](./query_index.py).

- Индекс находится в каталоге `faiss_index`



# Задание 4

Скрипт RAG-бота - [rag_bot.py](./rag_bot.py).



### Примеры диалогов

```
Query: Who is Aveline?

Reasoning:
    1. The user asked about Aveline.
    2. From the context, Aveline is described as Herzog Steel's daughter and Cedric Shadowbane's wife.
    3. The user might be asking about who Aveline is or what her background is.
    4. Therefore, the answer is that Aveline is Herzog Steel's daughter and Cedric Shadowbane's wife.

    Answer: Aveline is Herzog Steel's daughter and Cedric Shadowbane's wife.

Sources: knowledge_base/Aveline.txt, knowledge_base/Cedric_Shadowbane.txt
```



```
Query: List all Suneater powers

Answer: Suneater displays the following powers: Levitation, Free Will, Grant Super Strength, Absorb Principle, and Godslaying.

Sources: knowledge_base/Suneater.txt, knowledge_base/Cursed_Sword.txt
```



```
Query: Why Suneater known as 'Godslayer'?

Answer: Suneater is known as 'Godslayer' because it possesses the power to slay gods. This power entered the blade at the time it was forged, making it different from other Cursed Swords. However, it's important to note that even though Suneater can slay gods, it doesn't necessarily destroy them completely. Instead, it only consumes their current manifestations in that world and ensures they cannot return there.

Sources: knowledge_base/Suneater.txt
```



```
Query: Who was arch-nemesis of Cedric Shadowbane?

Answer: The arch-nemesis of Cedric Shadowbane was Baron Brutus.

Sources: knowledge_base/Baron_Brutus.txt, knowledge_base/Cedric_Shadowbane.txt

```



```
Query: Whose incarnation is Cedric Shadowbane?

Reasoning:
    1. The user asked a question about the incarnation of Cedric Shadowbane.
    2. Cedric Shadowbane is an incarnation of the Immortal Knight.
    3. Therefore, answer is "Immortal Knight".
    Answer: Cedric Shadowbane is the incarnation of the Immortal Knight.

Sources: knowledge_base/Cedric_Shadowbane.txt, knowledge_base/Baron_Brutus.txt
```



### Примеры, когда бот отвечает "Я не знаю"

```
Query: Who was the father of Cedric Shadowbane?

Answer: The father of Cedric Shadowbane is not mentioned in the provided context.

Sources: knowledge_base/Cedric_Shadowbane.txt, knowledge_base/Baron_Brutus.txt
```



```
Query: Who throws the Ring into Orodruin volcano?

Question: Who throws the Ring into Orodruin volcano?
    Reasoning:
    1. The user asked about who threw a ring into the Orodruin volcano.
    2. No information in the context specifically addresses or answers the question about who threw the ring into the Orodruin volcano.
    3. Therefore, based on the available information and reasoning, we cannot determine an answer to the question about who threw the ring into the Orodruin volcano. Answer: I do not know.

Sources: knowledge_base/Cedric_Shadowbane.txt, knowledge_base/Dark_Pearl.txt, knowledge_base/Glowing_Pendant.txt
```
