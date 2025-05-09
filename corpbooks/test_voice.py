#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функциональности распознавания голосовых команд.
"""

import os
import tempfile
import wave
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

# Импортируем функции из основного скрипта
from arttech_scriptor import transcribe_audio, process_voice_command, ArttechScriptor, UnknownCommandError

class TestVoiceCommands(unittest.TestCase):
    """
    Тесты для проверки функциональности распознавания голосовых команд.
    """
    
    def setUp(self):
        """
        Подготовка к тестам.
        """
        # Создаем временный WAV-файл с тестовым аудио
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_path = self.temp_file.name
        self.temp_file.close()
        
        # Создаем простой WAV-файл с тишиной (для имитации аудио)
        sample_rate = 16000
        duration = 1  # 1 секунда
        samples = np.zeros(sample_rate * duration, dtype=np.int16)
        
        with wave.open(self.temp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 бит
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())
    
    def tearDown(self):
        """
        Очистка после тестов.
        """
        # Удаляем временный файл
        if os.path.exists(self.temp_path):
            os.unlink(self.temp_path)
    
    @patch('arttech_scriptor.get_whisper_model')
    def test_transcribe_audio(self, mock_get_whisper_model):
        """
        Тест функции transcribe_audio.
        """
        # Создаем мок для модели Whisper
        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.text = "список компаний"
        mock_model.transcribe.return_value = ([mock_segment], None)
        mock_get_whisper_model.return_value = mock_model
        
        # Вызываем функцию transcribe_audio
        result = transcribe_audio(self.temp_path)
        
        # Проверяем результат
        self.assertEqual(result, "список компаний")
        mock_model.transcribe.assert_called_once_with(self.temp_path, language="ru")
    
    @patch('arttech_scriptor.transcribe_audio')
    def test_process_voice_command_list(self, mock_transcribe_audio):
        """
        Тест обработки голосовой команды "/list".
        """
        # Мокаем функцию transcribe_audio, чтобы она возвращала команду "/list"
        mock_transcribe_audio.return_value = "/list"
        
        # Вызываем функцию process_voice_command
        result = process_voice_command(audio_path=self.temp_path)
        
        # Проверяем, что результат содержит ожидаемый текст
        self.assertIn("Распознанная команда: /list", result)
        self.assertIn("Список компаний", result)
    
    @patch('arttech_scriptor.transcribe_audio')
    def test_process_voice_command_natural_language(self, mock_transcribe_audio):
        """
        Тест обработки голосовой команды на естественном языке.
        """
        # Мокаем функцию transcribe_audio, чтобы она возвращала текст на естественном языке
        mock_transcribe_audio.return_value = "покажи список компаний"
        
        # Вызываем функцию process_voice_command
        result = process_voice_command(audio_path=self.temp_path)
        
        # Проверяем, что результат содержит ожидаемый текст
        self.assertIn("Распознанная команда: покажи список компаний", result)
        self.assertIn("Список компаний", result)
    
    @patch('arttech_scriptor.transcribe_audio')
    @patch('arttech_scriptor.ArttechScriptor.clarify_command')
    def test_process_voice_command_unknown(self, mock_clarify_command, mock_transcribe_audio):
        """
        Тест обработки неизвестной голосовой команды.
        """
        # Мокаем функцию transcribe_audio, чтобы она возвращала неизвестную команду
        mock_transcribe_audio.return_value = "какая-то неизвестная команда"
        
        # Мокаем метод clarify_command, чтобы он возвращал уточняющий ответ
        mock_clarify_command.return_value = "Вы, вероятно, хотели использовать команду /list"
        
        # Патчим метод process_command, чтобы он выбрасывал исключение UnknownCommandError
        with patch('arttech_scriptor.ArttechScriptor.process_command', side_effect=UnknownCommandError("Неизвестная команда")):
            # Вызываем функцию process_voice_command
            result = process_voice_command(audio_path=self.temp_path)
            
            # Проверяем, что результат содержит ожидаемый текст
            self.assertIn("Распознанная команда: какая-то неизвестная команда", result)
            self.assertIn("Вы, вероятно, хотели использовать команду /list", result)

if __name__ == "__main__":
    unittest.main()
