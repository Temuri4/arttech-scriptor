#!/usr/bin/env python3
"""
Arttech Scriptor - Корпоративный ассистент для работы с базой корпоративных книг.

Обрабатывает команды:
- /list - список компаний
- /info <код> - информация из meta.json
- /get <код> <файл> - выдача файлов
- /ask [<код>] <вопрос> - GPT-поиск по базе
- /deadlines - ближайшие сроки
- /my - профиль пользователя

Поддерживает голосовой ввод команд:
- Через аудиофайл: python arttech_scriptor.py --voice file.wav
- Через микрофон: python arttech_scriptor.py --mic <длительность_в_секундах>
"""

import os
import json
import glob
import datetime
import pickle
import re
import logging
import numpy as np
import tempfile
import wave
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("arttech_scriptor")

class UnknownCommandError(Exception):
    """Исключение для неизвестных команд."""
    pass

def get_whisper_model():
    """
    Ленивая инициализация модели Whisper.
    
    Returns:
        Экземпляр модели Whisper
    """
    global _WHISPER_MODEL
    
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            
            # Инициализация модели Whisper (используем маленькую модель для быстрой работы)
            logger.info("Инициализация модели Whisper...")
            _WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
            logger.info("Модель Whisper успешно инициализирована")
        except ImportError:
            logger.error("Не удалось импортировать faster_whisper. Установите библиотеку: pip install faster-whisper")
            raise
    
    return _WHISPER_MODEL

def transcribe_audio(audio_path):
    """
    Преобразование аудиофайла в текст с помощью Whisper.
    
    Args:
        audio_path: Путь к аудиофайлу
        
    Returns:
        Распознанный текст
    """
    try:
        model = get_whisper_model()
        
        # Распознавание речи
        logger.info(f"Распознавание речи из файла: {audio_path}")
        segments, info = model.transcribe(audio_path, language="ru")
        
        # Объединение всех сегментов в один текст
        text = " ".join([segment.text for segment in segments])
        
        logger.info(f"Распознанный текст: {text}")
        return text.strip()
    except Exception as e:
        logger.error(f"Ошибка при распознавании речи: {e}")
        raise

def record_audio(duration=5, sample_rate=16000):
    """
    Запись аудио с микрофона.
    
    Args:
        duration: Длительность записи в секундах
        sample_rate: Частота дискретизации
        
    Returns:
        Путь к временному файлу с записанным аудио
    """
    try:
        import sounddevice as sd
        
        logger.info(f"Запись аудио с микрофона ({duration} сек)...")
        
        # Запись аудио
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Ожидание окончания записи
        
        # Создание временного файла для сохранения аудио
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Сохранение аудио в WAV-файл
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 бит
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        
        logger.info(f"Аудио записано в файл: {temp_path}")
        return temp_path
    except ImportError:
        logger.error("Не удалось импортировать sounddevice. Установите библиотеку: pip install sounddevice")
        raise
    except Exception as e:
        logger.error(f"Ошибка при записи аудио: {e}")
        raise

def process_voice_command(audio_path=None, mic_duration=None):
    """
    Обработка голосовой команды.
    
    Args:
        audio_path: Путь к аудиофайлу с командой
        mic_duration: Длительность записи с микрофона в секундах
        
    Returns:
        Результат выполнения команды
    """
    try:
        # Определяем источник аудио
        if audio_path:
            # Используем указанный аудиофайл
            path_to_process = audio_path
            temp_file = None
        elif mic_duration:
            # Записываем аудио с микрофона
            path_to_process = record_audio(duration=mic_duration)
            temp_file = path_to_process
        else:
            raise ValueError("Необходимо указать путь к аудиофайлу или длительность записи с микрофона")
        
        # Преобразуем аудио в текст
        text = transcribe_audio(path_to_process)
        
        # Удаляем временный файл, если он был создан
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
        
        # Если текст пустой, возвращаем сообщение об ошибке
        if not text:
            return "Не удалось распознать команду. Пожалуйста, повторите."
        
        # Создаем экземпляр ассистента
        assistant = ArttechScriptor()
        
        try:
            # Обрабатываем распознанный текст как команду
            result = assistant.process_command(text)
            return f"Распознанная команда: {text}\n\n{result}"
        except UnknownCommandError:
            clarification = assistant.clarify_command(text)
            return f"Распознанная команда: {text}\n\n{clarification}"
        except Exception as e:
            logger.error(f"Ошибка при обработке голосовой команды: {e}")
            return f"Произошла ошибка при обработке голосовой команды: {str(e)}"
    except Exception as e:
        logger.error(f"Ошибка при обработке голосовой команды: {e}")
        return f"Произошла ошибка при обработке голосовой команды: {str(e)}"


class ArttechScriptor:
    """
    Основной класс ассистента Arttech Scriptor для работы с корпоративными книгами.
    """
    
    def __init__(self, base_path: str = "/home/temo/HyperCyfra/projects/arttech-scriptor/corpbooks"):
        """
        Инициализация ассистента.
        
        Args:
            base_path: Базовый путь к директории с корпоративными книгами
        """
        self.base_path = Path(base_path)
        self.user_data = {}  # В будущем здесь будет храниться информация о пользователе
        self.embeddings_cache_path = self.base_path / ".embeddings_cache.pkl"
        self.embeddings_cache = self._load_embeddings_cache()
        
        # Проверка существования базовой директории
        if not self.base_path.exists():
            logger.error(f"Базовая директория {self.base_path} не существует")
            raise FileNotFoundError(f"Директория {self.base_path} не найдена")
    
    def _load_embeddings_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Загрузка кэша embeddings из файла.
        
        Returns:
            Словарь с кэшированными embeddings
        """
        if self.embeddings_cache_path.exists():
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш embeddings: {e}")
        return {}
    
    def _save_embeddings_cache(self) -> None:
        """
        Сохранение кэша embeddings в файл.
        """
        try:
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш embeddings: {e}")
    
    def get_companies(self) -> List[Dict[str, Any]]:
        """
        Получение списка всех компаний из базы.
        
        Returns:
            Список словарей с информацией о компаниях
        """
        companies = []
        
        # Поиск всех директорий компаний
        company_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        for company_dir in company_dirs:
            meta_path = company_dir / "meta.json"
            
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                        
                    companies.append({
                        "code": meta_data.get("company_code", company_dir.name),
                        "name": meta_data.get("name", company_dir.name),
                        "status": meta_data.get("status", "Неизвестно"),
                        "description": meta_data.get("short_description", "")
                    })
                except json.JSONDecodeError:
                    logger.error(f"Ошибка при чтении meta.json для {company_dir.name}")
                    companies.append({
                        "code": company_dir.name,
                        "name": company_dir.name,
                        "status": "Ошибка чтения meta.json",
                        "description": ""
                    })
            else:
                # Если meta.json отсутствует, добавляем базовую информацию
                companies.append({
                    "code": company_dir.name,
                    "name": company_dir.name,
                    "status": "Нет meta.json",
                    "description": ""
                })
        
        return companies
    
    def get_company_info(self, company_code: str) -> Dict[str, Any]:
        """
        Получение подробной информации о компании.
        
        Args:
            company_code: Код компании
            
        Returns:
            Словарь с информацией о компании
            
        Raises:
            FileNotFoundError: Если компания не найдена
        """
        company_dir = self.base_path / company_code
        meta_path = company_dir / "meta.json"
        
        if not company_dir.exists():
            raise FileNotFoundError(f"Компания с кодом {company_code} не найдена")
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Файл meta.json для компании {company_code} не найден")
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Дополнительная информация о структуре директории
            structure = {
                "raw_files": len(list((company_dir / "01_raw").glob("*"))) if (company_dir / "01_raw").exists() else 0,
                "processed_files": len(list((company_dir / "02_processed").glob("*"))) if (company_dir / "02_processed").exists() else 0,
                "assets": len(list((company_dir / "03_assets").glob("*"))) if (company_dir / "03_assets").exists() else 0
            }
            
            # Проверка наличия summary.md
            summary_exists = (company_dir / "summary.md").exists()
            
            # Объединение информации
            info = {
                **meta_data,
                "structure": structure,
                "summary_exists": summary_exists
            }
            
            return info
            
        except json.JSONDecodeError:
            raise ValueError(f"Ошибка при чтении meta.json для {company_code}")
    
    def get_file_content(self, company_code: str, file_path: str) -> Tuple[str, str]:
        """
        Получение содержимого файла.
        
        Args:
            company_code: Код компании
            file_path: Путь к файлу относительно директории компании
            
        Returns:
            Кортеж (содержимое файла, тип файла)
            
        Raises:
            FileNotFoundError: Если файл не найден
        """
        company_dir = self.base_path / company_code
        
        if not company_dir.exists():
            raise FileNotFoundError(f"Компания с кодом {company_code} не найдена")
        
        # Обработка относительного пути
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        full_path = company_dir / file_path
        
        # Проверка на попытку выхода за пределы директории компании
        if not str(full_path).startswith(str(company_dir)):
            raise ValueError(f"Недопустимый путь к файлу: {file_path}")
        
        if not full_path.exists():
            raise FileNotFoundError(f"Файл {file_path} не найден для компании {company_code}")
        
        # Определение типа файла по расширению
        file_extension = full_path.suffix.lower()
        
        # Текстовые форматы, которые можно прочитать
        text_extensions = ['.txt', '.md', '.json', '.csv', '.html', '.xml', '.py', '.js', '.css']
        
        if file_extension in text_extensions:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, file_extension[1:]  # Возвращаем расширение без точки
            except UnicodeDecodeError:
                return f"Файл {file_path} не может быть прочитан как текст", file_extension[1:]
        else:
            return f"Файл {file_path} имеет бинарный формат и не может быть отображен как текст", file_extension[1:]
    
    def _extract_company_and_query(self, query_text: str) -> Tuple[Optional[str], str]:
        """
        Извлечение кода компании и текста запроса из строки запроса.
        
        Args:
            query_text: Текст запроса
            
        Returns:
            Кортеж (код компании или None, текст запроса)
        """
        # Получаем список всех компаний
        companies = self.get_companies()
        company_codes = [company["code"] for company in companies]
        
        # Разбиваем запрос на слова
        words = query_text.split()
        
        if not words:
            return None, query_text
        
        # Проверяем, является ли первое слово кодом компании
        first_word = words[0].upper()
        
        if first_word in company_codes:
            # Если первое слово - код компании, возвращаем его и остальной запрос
            return first_word, " ".join(words[1:])
        
        # Если код компании не найден, возвращаем None и исходный запрос
        return None, query_text
    
    def _get_text_files(self, company_code: Optional[str] = None) -> List[Tuple[str, Path]]:
        """
        Получение списка всех текстовых файлов для указанной компании или всех компаний.
        
        Args:
            company_code: Код компании или None для всех компаний
            
        Returns:
            Список кортежей (код компании, путь к файлу)
        """
        text_files = []
        text_extensions = ['.txt', '.md', '.json', '.csv', '.html', '.xml', '.py', '.js', '.css']
        
        if company_code:
            # Если указан код компании, ищем файлы только в этой директории
            company_dir = self.base_path / company_code
            if not company_dir.exists():
                logger.warning(f"Компания с кодом {company_code} не найдена")
                return []
            
            # Рекурсивный поиск всех файлов в директории компании
            for ext in text_extensions:
                for file_path in company_dir.glob(f"**/*{ext}"):
                    if file_path.is_file():
                        text_files.append((company_code, file_path))
        else:
            # Если код компании не указан, ищем файлы во всех компаниях
            companies = self.get_companies()
            
            for company in companies:
                company_code = company["code"]
                company_dir = self.base_path / company_code
                
                # Рекурсивный поиск всех файлов в директории компании
                for ext in text_extensions:
                    for file_path in company_dir.glob(f"**/*{ext}"):
                        if file_path.is_file():
                            text_files.append((company_code, file_path))
        
        return text_files
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Разбиение текста на перекрывающиеся чанки.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в символах
            overlap: Размер перекрытия между чанками в символах
            
        Returns:
            Список чанков текста
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Определяем конец текущего чанка
            end = min(start + chunk_size, len(text))
            
            # Если это не последний чанк и не достигли конца текста,
            # пытаемся найти конец предложения или абзаца для более естественного разделения
            if end < len(text):
                # Ищем ближайший конец предложения или абзаца после end - overlap
                sentence_end = max(
                    text.rfind(". ", end - overlap, end),
                    text.rfind(".\n", end - overlap, end),
                    text.rfind("\n\n", end - overlap, end)
                )
                
                if sentence_end != -1:
                    # Если нашли подходящее место для разделения, используем его
                    end = sentence_end + 1  # +1 чтобы включить точку или перевод строки
            
            # Добавляем чанк в список
            chunks.append(text[start:end])
            
            # Перемещаем начало следующего чанка с учетом перекрытия
            start = end - overlap if end - overlap > start else end
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Получение embedding для текста с использованием OpenAI API.
        
        Args:
            text: Текст для получения embedding
            
        Returns:
            Список чисел с плавающей точкой (embedding)
            
        Raises:
            Exception: Если произошла ошибка при получении embedding
        """
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API недоступен или отсутствует API ключ")
        
        try:
            # Используем новое API для получения embedding
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            # Извлекаем embedding из ответа
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"Ошибка при получении embedding: {e}")
            raise
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Вычисление косинусного сходства между двумя векторами.
        
        Args:
            a: Первый вектор
            b: Второй вектор
            
        Returns:
            Косинусное сходство (от -1 до 1)
        """
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 0
        
        return np.dot(a, b) / (a_norm * b_norm)
    
    def _get_file_chunks_with_embeddings(self, company_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение чанков текста из файлов с их embeddings.
        
        Args:
            company_code: Код компании или None для всех компаний
            
        Returns:
            Список словарей с информацией о чанках и их embeddings
        """
        if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
            logger.warning("OpenAI API недоступен или отсутствует API ключ. Возвращаем пустой список чанков.")
            return []
        
        chunks_with_embeddings = []
        text_files = self._get_text_files(company_code)
        
        for company_code, file_path in text_files:
            # Формируем относительный путь для использования в кэше
            rel_path = str(file_path.relative_to(self.base_path))
            cache_key = f"{company_code}:{rel_path}"
            
            # Проверяем, есть ли файл в кэше и не изменился ли он
            file_mtime = file_path.stat().st_mtime
            
            if cache_key in self.embeddings_cache and self.embeddings_cache[cache_key]["mtime"] == file_mtime:
                # Если файл в кэше и не изменился, используем кэшированные данные
                chunks_with_embeddings.extend(self.embeddings_cache[cache_key]["chunks"])
                logger.debug(f"Использованы кэшированные данные для {cache_key}")
                continue
            
            # Если файла нет в кэше или он изменился, обрабатываем его
            try:
                # Читаем содержимое файла
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Разбиваем текст на чанки
                chunks = self._chunk_text(content)
                
                # Создаем список чанков с их embeddings
                file_chunks = []
                
                for i, chunk in enumerate(chunks):
                    # Получаем embedding для чанка
                    embedding = self._get_embedding(chunk)
                    
                    # Добавляем информацию о чанке
                    file_chunks.append({
                        "company_code": company_code,
                        "file_path": str(file_path),
                        "chunk_index": i,
                        "content": chunk,
                        "embedding": embedding
                    })
                
                # Добавляем чанки в общий список
                chunks_with_embeddings.extend(file_chunks)
                
                # Обновляем кэш
                self.embeddings_cache[cache_key] = {
                    "mtime": file_mtime,
                    "chunks": file_chunks
                }
                
                # Сохраняем кэш после обработки каждого файла
                self._save_embeddings_cache()
                
                logger.debug(f"Обработан файл {cache_key}, получено {len(file_chunks)} чанков")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        
        return chunks_with_embeddings
    
    def _find_relevant_chunks(self, query: str, company_code: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Поиск наиболее релевантных чанков текста для запроса.
        
        Args:
            query: Текст запроса
            company_code: Код компании или None для поиска по всем компаниям
            top_k: Количество наиболее релевантных чанков для возврата
            
        Returns:
            Список словарей с информацией о наиболее релевантных чанках
        """
        if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
            logger.warning("OpenAI API недоступен или отсутствует API ключ. Возвращаем пустой список релевантных чанков.")
            return []
        
        try:
            # Получаем embedding для запроса
            query_embedding = self._get_embedding(query)
            
            # Получаем все чанки с их embeddings
            all_chunks = self._get_file_chunks_with_embeddings(company_code)
            
            if not all_chunks:
                logger.warning("Не найдено чанков для поиска")
                return []
            
            # Вычисляем косинусное сходство между запросом и каждым чанком
            for chunk in all_chunks:
                chunk["similarity"] = self._cosine_similarity(query_embedding, chunk["embedding"])
            
            # Сортируем чанки по убыванию сходства
            sorted_chunks = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)
            
            # Возвращаем top_k наиболее релевантных чанков
            return sorted_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка при поиске релевантных чанков: {e}")
            return []
    
    def ask(self, query: str) -> str:
        """
        Обработка запроса пользователя с использованием GPT или простого поиска.
        
        Args:
            query: Текст запроса
            
        Returns:
            Ответ на запрос
        """
        # Извлекаем код компании и текст запроса
        company_code, query_text = self._extract_company_and_query(query)
        
        # Если запрос пустой, возвращаем сообщение об ошибке
        if not query_text.strip():
            return "Пожалуйста, укажите текст запроса после кода компании."
        
        # Если доступен OpenAI API, используем его
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            return self._ask_with_gpt(query_text, company_code)
        else:
            # Иначе используем простой поиск по ключевым словам
            return self._simple_search(query_text, company_code)
    
    def _ask_with_gpt(self, query: str, company_code: Optional[str] = None) -> str:
        """
        Обработка запроса с использованием GPT.
        
        Args:
            query: Текст запроса
            company_code: Код компании или None для поиска по всем компаниям
            
        Returns:
            Ответ на запрос
        """
        try:
            # Находим наиболее релевантные чанки для запроса
            relevant_chunks = self._find_relevant_chunks(query, company_code, top_k=8)
            
            if not relevant_chunks:
                if company_code:
                    return f"Не удалось найти релевантную информацию для запроса в компании {company_code}."
                else:
                    return "Не удалось найти релевантную информацию для запроса."
            
            # Формируем контекст из релевантных чанков
            context = ""
            sources = set()
            
            for i, chunk in enumerate(relevant_chunks):
                context += f"Фрагмент {i+1} (из {chunk['file_path']}):\n{chunk['content']}\n\n"
                sources.add(chunk['file_path'])
            
            # Формируем запрос к GPT
            system_prompt = """
            Вы ассистент Arttech Scriptor, который помогает с информацией о корпоративных книгах.
            Отвечайте на вопросы пользователя, основываясь только на предоставленном контексте.
            Если в контексте нет информации для ответа на вопрос, честно признайтесь в этом.
            Ваши ответы должны быть информативными, точными и хорошо структурированными.
            """
            
            user_prompt = f"""
            Контекст:
            {context}
            
            Вопрос: {query}
            """
            
            # Отправляем запрос к GPT
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",  # Используем модель с большим контекстным окном
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            # Извлекаем ответ
            answer = response.choices[0].message.content
            
            # Добавляем информацию об источниках
            sources_info = "\n\nИсточники:\n" + "\n".join([f"- {src}" for src in sources])
            
            return answer + sources_info
            
        except Exception as e:
            logger.error(f"Ошибка при запросе к GPT: {e}")
            return f"Произошла ошибка при обработке запроса через GPT: {e}. Попробуйте использовать простой поиск."
    
    def _simple_search(self, query: str, company_code: Optional[str] = None) -> str:
        """
        Простой поиск по ключевым словам.
        
        Args:
            query: Текст запроса
            company_code: Код компании или None для поиска по всем компаниям
            
        Returns:
            Результат поиска
        """
        results = []
        query_terms = query.lower().split()
        
        # Получаем список компаний для поиска
        if company_code:
            companies = [{"code": company_code}]
        else:
            companies = self.get_companies()
        
        for company in companies:
            company_code = company["code"]
            company_dir = self.base_path / company_code
            
            # Поиск в meta.json
            meta_path = company_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    meta_text = json.dumps(meta_data, ensure_ascii=False).lower()
                    
                    # Проверяем, содержит ли meta.json все термины запроса
                    if all(term in meta_text for term in query_terms):
                        results.append(f"Найдено в meta.json компании {meta_data.get('name', company_code)} ({company_code})")
                except Exception as e:
                    logger.error(f"Ошибка при чтении meta.json для {company_code}: {e}")
            
            # Поиск в summary.md
            summary_path = company_dir / "summary.md"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary_content = f.read().lower()
                    
                    # Проверяем, содержит ли summary.md все термины запроса
                    if all(term in summary_content for term in query_terms):
                        results.append(f"Найдено в summary.md компании {company_code}")
                        
                        # Находим контекст вокруг первого вхождения первого термина
                        term = query_terms[0]
                        pos = summary_content.find(term)
                        if pos >= 0:
                            start = max(0, pos - 100)
                            end = min(len(summary_content), pos + 100)
                            context = summary_content[start:end]
                            results.append(f"Контекст: ...{context}...")
                except Exception as e:
                    logger.error(f"Ошибка при чтении summary.md для {company_code}: {e}")
            
            # Поиск в других текстовых файлах
            for subdir in ["01_raw", "02_processed"]:
                subdir_path = company_dir / subdir
                if subdir_path.exists():
                    # Используем list() для преобразования генераторов в списки перед объединением
                    text_files = list(subdir_path.glob("**/*.txt")) + list(subdir_path.glob("**/*.md"))
                    for file_path in text_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read().lower()
                            
                            # Проверяем, содержит ли файл все термины запроса
                            if all(term in file_content for term in query_terms):
                                rel_path = file_path.relative_to(company_dir)
                                results.append(f"Найдено в файле {rel_path} компании {company_code}")
                                
                                # Находим контекст вокруг первого вхождения первого термина
                                term = query_terms[0]
                                pos = file_content.find(term)
                                if pos >= 0:
                                    start = max(0, pos - 100)
                                    end = min(len(file_content), pos + 100)
                                    context = file_content[start:end]
                                    results.append(f"Контекст: ...{context}...")
                        except Exception as e:
                            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
        
        if results:
            return "\n".join(results)
        else:
            if company_code:
                return f"По вашему запросу ничего не найдено в компании {company_code}."
            else:
                return "По вашему запросу ничего не найдено."
    
    def search_content(self, query: str) -> str:
        """
        Поиск информации по базе данных с использованием GPT или простого поиска.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Результат поиска
        """
        return self.ask(query)
    
    def get_deadlines(self) -> List[Dict[str, Any]]:
        """
        Получение ближайших дедлайнов по всем компаниям.
        
        Returns:
            Список словарей с информацией о дедлайнах
        """
        all_deadlines = []
        
        # Получаем список всех компаний
        companies = self.get_companies()
        
        for company in companies:
            company_code = company["code"]
            company_dir = self.base_path / company_code
            meta_path = company_dir / "meta.json"
            
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    # Извлекаем дедлайны
                    deadlines = meta_data.get("deadlines", {})
                    
                    for deadline_name, deadline_date_str in deadlines.items():
                        try:
                            deadline_date = datetime.strptime(deadline_date_str, "%Y-%m-%d")
                            
                            all_deadlines.append({
                                "company_code": company_code,
                                "company_name": meta_data.get("name", company_code),
                                "deadline_name": deadline_name,
                                "deadline_date": deadline_date,
                                "days_left": (deadline_date - datetime.now()).days
                            })
                        except ValueError:
                            logger.error(f"Неверный формат даты для дедлайна {deadline_name} компании {company_code}")
                
                except Exception as e:
                    logger.error(f"Ошибка при чтении meta.json для {company_code}: {e}")
        
        # Сортируем дедлайны по дате (ближайшие сначала)
        all_deadlines.sort(key=lambda x: x["deadline_date"])
        
        return all_deadlines
    
    def get_user_profile(self, user_id: str = "default") -> Dict[str, Any]:
        """
        Получение профиля пользователя.
        
        Args:
            user_id: Идентификатор пользователя
            
        Returns:
            Словарь с информацией о пользователе
        """
        # TODO: В будущем реализовать полноценную систему пользователей
        # Пока возвращаем заглушку
        
        return {
            "user_id": user_id,
            "name": "Пользователь Arttech Scriptor",
            "role": "Редактор",
            "assigned_companies": ["NIIPH", "ASIZ"],
            "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def clarify_command(self, text: str) -> str:
        """
        Уточнение неясной команды с использованием GPT.
        
        Args:
            text: Распознанный текст
            
        Returns:
            Уточняющий ответ
        """
        try:
            # Формируем запрос к GPT
            system_prompt = """
            Вы ассистент Arttech Scriptor, который помогает пользователям с командами.
            Доступные команды:
            - /list - список компаний
            - /info <код> - информация из meta.json
            - /get <код> <файл> - выдача файлов
            - /ask [<код>] <вопрос> - GPT-поиск по базе
            - /deadlines - ближайшие сроки
            - /my - профиль пользователя
            
            Пользователь произнес фразу, которая не была распознана как команда.
            Ваша задача - определить, какую команду пользователь, вероятно, хотел использовать,
            и предложить правильный вариант команды.
            """
            
            user_prompt = f"""
            Пользователь произнес: "{text}"
            
            Какую команду он, вероятно, хотел использовать? Предложите правильный вариант.
            """
            
            # Отправляем запрос к GPT
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            # Извлекаем ответ
            answer = response.choices[0].message.content
            
            return f"Не удалось распознать команду: '{text}'.\n\n{answer}"
            
        except Exception as e:
            logger.error(f"Ошибка при уточнении команды: {e}")
            return f"Не удалось распознать команду: '{text}'. Доступные команды: /list, /info, /get, /ask, /deadlines, /my"
    
    def process_command(self, command: str) -> str:
        """
        Обработка команды от пользователя.
        
        Args:
            command: Команда пользователя
            
        Returns:
            Результат выполнения команды
            
        Raises:
            UnknownCommandError: Если команда не распознана
        """
        # Удаляем лишние пробелы
        command = command.strip()
        
        # Проверяем, начинается ли команда с "/"
        if not command.startswith("/"):
            # Проверяем, содержит ли текст ключевые слова, которые могут указывать на команду
            command_lower = command.lower()
            
            if "список" in command_lower or "компании" in command_lower:
                return self.process_command("/list")
            elif "информация" in command_lower and any(company["code"].lower() in command_lower for company in self.get_companies()):
                # Извлекаем код компании
                for company in self.get_companies():
                    if company["code"].lower() in command_lower:
                        return self.process_command(f"/info {company['code']}")
            elif "файл" in command_lower and any(company["code"].lower() in command_lower for company in self.get_companies()):
                # Это может быть команда /get, но нужно больше информации
                pass
            elif "вопрос" in command_lower or "спросить" in command_lower:
                # Преобразуем в команду /ask
                return self.process_command(f"/ask {command}")
            elif "дедлайн" in command_lower or "срок" in command_lower:
                return self.process_command("/deadlines")
            elif "профиль" in command_lower or "пользователь" in command_lower:
                return self.process_command("/my")
            
            # Если не удалось распознать команду, выбрасываем исключение
            raise UnknownCommandError(f"Команда должна начинаться с символа '/'. Доступные команды: /list, /info, /get, /ask, /deadlines, /my")
        
        # Разбиваем команду на части
        parts = command.split()
        cmd = parts[0].lower()
        
        try:
            # Обработка команды /list
            if cmd == "/list":
                companies = self.get_companies()
                if not companies:
                    return "В базе данных нет компаний."
                
                result = "Список компаний:\n\n"
                for company in companies:
                    result += f"📁 {company['code']} - {company['name']}\n"
                    result += f"   Статус: {company['status']}\n"
                    if company['description']:
                        result += f"   {company['description']}\n"
                    result += "\n"
                
                return result
            
            # Обработка команды /info
            elif cmd == "/info":
                if len(parts) < 2:
                    return "Необходимо указать код компании. Пример: /info NIIPH"
                
                company_code = parts[1].upper()
                info = self.get_company_info(company_code)
                
                result = f"📊 Информация о компании {info.get('name', company_code)} ({company_code}):\n\n"
                result += f"Статус: {info.get('status', 'Неизвестно')}\n"
                result += f"Описание: {info.get('short_description', '')}\n"
                
                if 'keywords' in info and info['keywords']:
                    result += f"Ключевые слова: {', '.join(info['keywords'])}\n"
                
                if 'deadlines' in info and info['deadlines']:
                    result += "\nДедлайны:\n"
                    for name, date in info['deadlines'].items():
                        result += f"- {name}: {date}\n"
                
                if 'structure' in info:
                    result += "\nСтруктура:\n"
                    result += f"- Сырые файлы: {info['structure']['raw_files']}\n"
                    result += f"- Обработанные файлы: {info['structure']['processed_files']}\n"
                    result += f"- Ассеты: {info['structure']['assets']}\n"
                
                if info.get('summary_exists', False):
                    result += "\nДоступна сводка (summary.md). Используйте /get для просмотра.\n"
                
                if 'last_updated' in info:
                    result += f"\nПоследнее обновление: {info['last_updated']}\n"
                
                return result
            
            # Обработка команды /get
            elif cmd == "/get":
                if len(parts) < 3:
                    return "Необходимо указать код компании и путь к файлу. Пример: /get NIIPH summary.md"
                
                company_code = parts[1].upper()
                file_path = " ".join(parts[2:])  # Объединяем все оставшиеся части как путь к файлу
                
                content, file_type = self.get_file_content(company_code, file_path)
                
                result = f"📄 Файл: {file_path} ({file_type})\n"
                result += f"Компания: {company_code}\n\n"
                result += "```\n"
                result += content
                result += "\n```"
                
                return result
            
            # Обработка команды /ask
            elif cmd == "/ask":
                if len(parts) < 2:
                    return "Необходимо указать вопрос. Пример: /ask Какие компании связаны с химической промышленностью?"
                
                query = " ".join(parts[1:])
                result = self.ask(query)
                
                return f"🔍 Результаты поиска по запросу: '{query}'\n\n{result}"
            
            # Обработка команды /deadlines
            elif cmd == "/deadlines":
                deadlines = self.get_deadlines()
                
                if not deadlines:
                    return "Дедлайны не найдены."
                
                result = "⏰ Ближайшие дедлайны:\n\n"
                
                # Группируем дедлайны по компаниям
                deadlines_by_company = {}
                for deadline in deadlines:
                    company_code = deadline["company_code"]
                    if company_code not in deadlines_by_company:
                        deadlines_by_company[company_code] = []
                    deadlines_by_company[company_code].append(deadline)
                
                # Выводим дедлайны по компаниям
                for company_code, company_deadlines in deadlines_by_company.items():
                    company_name = company_deadlines[0]["company_name"]
                    result += f"📁 {company_name} ({company_code}):\n"
                    
                    for deadline in company_deadlines:
                        days_left = deadline["days_left"]
                        date_str = deadline["deadline_date"].strftime("%d.%m.%Y")
                        
                        # Добавляем эмодзи в зависимости от срочности
                        if days_left < 0:
                            emoji = "🔴"  # Просрочено
                        elif days_left < 7:
                            emoji = "🟠"  # Менее недели
                        elif days_left < 30:
                            emoji = "🟡"  # Менее месяца
                        else:
                            emoji = "🟢"  # Более месяца
                        
                        if days_left < 0:
                            days_text = f"просрочено на {abs(days_left)} дн."
                        else:
                            days_text = f"осталось {days_left} дн."
                        
                        result += f"  {emoji} {deadline['deadline_name']}: {date_str} ({days_text})\n"
                    
                    result += "\n"
                
                return result
            
            # Обработка команды /my
            elif cmd == "/my":
                profile = self.get_user_profile()
                
                result = "👤 Профиль пользователя:\n\n"
                result += f"Имя: {profile['name']}\n"
                result += f"Роль: {profile['role']}\n"
                result += f"Назначенные компании: {', '.join(profile['assigned_companies'])}\n"
                result += f"Последняя активность: {profile['last_activity']}\n"
                
                # TODO: В будущем добавить статистику работы пользователя
                
                return result
            
            # Неизвестная команда
            else:
                raise UnknownCommandError(f"Неизвестная команда: {cmd}. Доступные команды: /list, /info, /get, /ask, /deadlines, /my")
        
        except FileNotFoundError as e:
            return f"Ошибка: {str(e)}"
        except ValueError as e:
            return f"Ошибка: {str(e)}"
        except Exception as e:
            logger.error(f"Необработанная ошибка: {e}")
            return f"Произошла ошибка при выполнении команды: {str(e)}"


# Если скрипт запущен напрямую, а не импортирован
if __name__ == "__main__":
    import sys
    import argparse
    
    # Создаем парсер аргументов командной строки
    parser = argparse.ArgumentParser(description="Arttech Scriptor - Корпоративный ассистент для работы с базой корпоративных книг")
    parser.add_argument("--voice", type=str, help="Путь к аудиофайлу с голосовой командой")
    parser.add_argument("--mic", type=int, help="Длительность записи с микрофона в секундах")
    parser.add_argument("command", nargs="*", help="Команда для выполнения")
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Обработка голосовой команды
    if args.voice or args.mic is not None:
        result = process_voice_command(audio_path=args.voice, mic_duration=args.mic)
        print(result)
    # Обработка текстовой команды из аргументов
    elif args.command:
        # Создаем экземпляр ассистента
        assistant = ArttechScriptor()
        command = " ".join(args.command)
        result = assistant.process_command(command)
        print(result)
    else:
        # Интерактивный режим
        print("Arttech Scriptor - Ассистент для работы с корпоративными книгами")
        print("Введите команду или 'exit' для выхода")
        print("Для голосового ввода используйте --voice <путь_к_файлу> или --mic <длительность>")
        
        # Создаем экземпляр ассистента
        assistant = ArttechScriptor()
        
        while True:
            command = input("> ")
            
            if command.lower() in ["exit", "quit", "выход"]:
                break
            
            try:
                result = assistant.process_command(command)
                print(result)
            except UnknownCommandError:
                # Если команда не распознана, используем GPT для уточнения
                clarification = assistant.clarify_command(command)
                print(clarification)
