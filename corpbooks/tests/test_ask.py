#!/usr/bin/env python3
"""
Тесты для команды /ask в ArttechScriptor
"""

import os
import sys
import unittest
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта
sys.path.append(str(Path(__file__).parent.parent))

from arttech_scriptor import ArttechScriptor

class TestAskCommand(unittest.TestCase):
    """
    Тесты для команды /ask
    """
    
    def setUp(self):
        """
        Подготовка к тестам
        """
        self.scriptor = ArttechScriptor()
    
    def test_ask_with_company(self):
        """
        Тест команды /ask с указанием компании
        """
        # Проверяем, что команда /ask с указанием компании возвращает непустой результат
        result = self.scriptor.process_command("/ask NIIPH инновации")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        self.assertIn("NIIPH", result)
    
    def test_ask_without_company(self):
        """
        Тест команды /ask без указания компании
        """
        # Проверяем, что команда /ask без указания компании возвращает непустой результат
        result = self.scriptor.process_command("/ask инновации в промышленности")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
    
    def test_extract_company_and_query(self):
        """
        Тест метода _extract_company_and_query
        """
        # Проверяем корректное извлечение компании и запроса
        company, query = self.scriptor._extract_company_and_query("NIIPH какие инновации используются?")
        self.assertEqual(company, "NIIPH")
        self.assertEqual(query, "какие инновации используются?")
        
        # Проверяем случай без указания компании
        company, query = self.scriptor._extract_company_and_query("какие инновации используются?")
        self.assertIsNone(company)
        self.assertEqual(query, "какие инновации используются?")
    
    def test_simple_search(self):
        """
        Тест метода _simple_search
        """
        # Проверяем поиск по конкретной компании
        result = self.scriptor._simple_search("инновации", "NIIPH")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        
        # Проверяем поиск по всем компаниям
        result = self.scriptor._simple_search("инновации")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")


if __name__ == "__main__":
    unittest.main()
