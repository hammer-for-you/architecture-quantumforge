import json
import os
import re
from pathlib import Path
from typing import Dict, List


def load_replacements_map(json_file_path: str) -> List[Dict[str, str]]:
    """
    Загружает карту замен из JSON файла.
    
    Args:
        json_file_path (str): Путь к JSON файлу с картой замен
        
    Returns:
        List[Dict[str, str]]: Список словарей с заменами в исходном порядке
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            replacements = json.load(file)
            print(f"Загружено {len(replacements)} замен из {json_file_path}")
            return replacements
    except FileNotFoundError:
        print(f"Ошибка: Файл {json_file_path} не найден")
        raise
    except json.JSONDecodeError:
        print(f"Ошибка: Неверный формат JSON в файле {json_file_path}")
        raise


def process_files(input_dir: str, output_dir: str, replacements: List[Dict[str, str]]):
    """
    Обрабатывает все текстовые файлы в каталоге, применяя замены в исходном порядке.
    
    Args:
        input_dir (str): Входной каталог с исходными файлами
        output_dir (str): Выходной каталог для обработанных файлов
        replacements (List[Dict[str, str]]): Список замен для применения (сохраняется порядок)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    text_files = []
    for ext in ['*.txt', '*.md', '*.csv', '*.json', '*.xml', '*.html', '*.htm']:
        text_files.extend(Path(input_dir).glob(ext))
    
    print(f"Найдено {len(text_files)} текстовых файлов для обработки")
    
    processed_count = 0
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            for replacement in replacements:
                from_text = replacement['from']
                to_text = replacement['to']
                
                pattern = re.escape(from_text)
                content = re.sub(pattern, to_text, content)
            
            output_file_path = Path(output_dir) / file_path.name
            
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            processed_count += 1
            print(f"Обработан файл: {file_path.name}")
            
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")
    
    print(f"Обработка завершена. Обработано файлов: {processed_count}")


def main():
    """
    Основная функция скрипта.
    """
    json_file_path = "json/terms_map.json"
    input_directory = "raw_data"
    output_directory = "knowledge_base"
    
    print("Запуск скрипта замены текста...")
    print("Замены применяются в порядке, указанном в массиве JSON")
    
    try:
        replacements = load_replacements_map(json_file_path)
        
        process_files(input_directory, output_directory, replacements)
        
        print("Скрипт успешно завершил работу!")
        
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())