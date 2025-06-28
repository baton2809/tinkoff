"""
Демонстрация использования обученной Reward Model
Показывает как загрузить и использовать reward model для оценки качества ответов
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from typing import List, Tuple

class RewardModelEvaluator:
    """
    Класс для работы с обученной reward model
    """
    
    def __init__(self, model_path: str = "./reward_model_output"):
        """
        Инициализация evaluator'а
        
        Args:
            model_path: Путь к обученной reward model
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        self.load_model()
    
    def _get_device(self):
        """Определяет доступное устройство"""
        if torch.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Загружает обученную reward model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена в {self.model_path}. Сначала обучите модель.")
        
        print(f"Загружаем reward model из: {self.model_path}")
        print(f"Устройство: {self.device}")
        
        # Загружаем модель и токенизатор
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Переносим модель на устройство
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Reward model загружена")
        print(f"✓ Количество параметров: {self.model.num_parameters():,}")
    
    def get_reward_score(self, prompt: str, response: str, max_length: int = 256) -> float:
        """
        Получает reward score для пары prompt-response
        
        Args:
            prompt: Исходный запрос
            response: Ответ на запрос
            max_length: Максимальная длина токенов
            
        Returns:
            Reward score (float)
        """
        # Комбинируем prompt и response
        text = f"{prompt}\n{response}"
        
        # Токенизируем
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Переносим на устройство
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Получаем reward score
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward_score = outputs.logits.squeeze().item()
        
        return reward_score
    
    def compare_responses(self, prompt: str, responses: List[str]) -> List[Tuple[str, float]]:
        """
        Сравнивает несколько ответов на один prompt
        
        Args:
            prompt: Исходный запрос
            responses: Список ответов для сравнения
            
        Returns:
            Список кортежей (response, reward_score), отсортированный по убыванию score
        """
        results = []
        
        for response in responses:
            score = self.get_reward_score(prompt, response)
            results.append((response, score))
        
        # Сортируем по убыванию reward score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def evaluate_conversation(self, conversation: List[Tuple[str, str]]) -> List[float]:
        """
        Оценивает качество диалога
        
        Args:
            conversation: Список кортежей (prompt, response)
            
        Returns:
            Список reward scores для каждого обмена
        """
        scores = []
        
        for prompt, response in conversation:
            score = self.get_reward_score(prompt, response)
            scores.append(score)
        
        return scores

def demo_basic_usage():
    """
    Демонстрация базового использования reward model
    """
    print("=== Демонстрация базового использования Reward Model ===\n")
    
    # Создаем evaluator
    evaluator = RewardModelEvaluator()
    
    # Тестовые примеры
    test_cases = [
        {
            "prompt": "Объясни, что такое машинное обучение",
            "responses": [
                "Машинное обучение - это область искусственного интеллекта, которая позволяет компьютерам учиться и принимать решения на основе данных без явного программирования.",
                "Это когда компьютер учится сам.",
                "Не знаю, что это такое.",
                "Машинное обучение включает алгоритмы, которые автоматически улучшают свою производительность через опыт. Основные типы: обучение с учителем, без учителя и обучение с подкреплением."
            ]
        },
        {
            "prompt": "Как приготовить омлет?",
            "responses": [
                "Взбейте 2-3 яйца, добавьте соль и перец. Разогрейте сковороду с маслом, вылейте яйца и готовьте на среднем огне 2-3 минуты.",
                "Разбейте яйца в сковороду.",
                "Купите готовый омлет в магазине.",
                "Возьмите яйца, взбейте их с молоком, солью и специями. Готовьте на сковороде с антипригарным покрытием до готовности."
            ]
        }
    ]
    
    # Тестируем каждый случай
    for i, test_case in enumerate(test_cases, 1):
        print(f"Тест {i}: {test_case['prompt']}")
        print("-" * 50)
        
        # Сравниваем ответы
        results = evaluator.compare_responses(test_case['prompt'], test_case['responses'])
        
        # Показываем результаты
        for rank, (response, score) in enumerate(results, 1):
            print(f"Ранг {rank} (score: {score:.4f}): {response[:100]}{'...' if len(response) > 100 else ''}")
        
        print("\n")

def demo_conversation_evaluation():
    """
    Демонстрация оценки диалога
    """
    print("=== Демонстрация оценки диалога ===\n")
    
    evaluator = RewardModelEvaluator()
    
    # Пример диалога
    conversation = [
        ("Привет! Как дела?", "Привет! Дела хорошо, спасибо! Как у тебя?"),
        ("Что ты умеешь делать?", "Я могу помочь с различными задачами: отвечать на вопросы, объяснять концепции, помогать с программированием и многое другое."),
        ("Расскажи про Python", "Python - это высокоуровневый язык программирования, известный своей простотой и читаемостью. Он широко используется в веб-разработке, анализе данных, машинном обучении и автоматизации."),
        ("Спасибо!", "Пожалуйста! Обращайтесь, если будут еще вопросы.")
    ]
    
    # Оцениваем диалог
    scores = evaluator.evaluate_conversation(conversation)
    
    print("Оценка диалога:")
    print("-" * 50)
    
    total_score = 0
    for i, ((prompt, response), score) in enumerate(zip(conversation, scores), 1):
        print(f"Обмен {i} (score: {score:.4f}):")
        print(f"  Пользователь: {prompt}")
        print(f"  Ассистент: {response}")
        print()
        total_score += score
    
    avg_score = total_score / len(scores)
    print(f"Средний score диалога: {avg_score:.4f}")

def demo_response_ranking():
    """
    Демонстрация ранжирования ответов
    """
    print("=== Демонстрация ранжирования ответов ===\n")
    
    evaluator = RewardModelEvaluator()
    
    prompt = "Объясни разницу между списком и кортежем в Python"
    
    responses = [
        "Список изменяемый, кортеж нет.",
        "Список (list) в Python - это изменяемая структура данных, которая может содержать элементы разных типов. Кортеж (tuple) - неизменяемая структура данных. Списки используют квадратные скобки [], кортежи - круглые ().",
        "Не знаю разницы.",
        "Основные различия: 1) Списки изменяемы (mutable), кортежи неизменяемы (immutable). 2) Списки используют [], кортежи (). 3) Кортежи быстрее для чтения. 4) Кортежи можно использовать как ключи словаря.",
        "Список можно менять, а кортеж нельзя. Список пишется в [], кортеж в ()."
    ]
    
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    # Ранжируем ответы
    ranked_responses = evaluator.compare_responses(prompt, responses)
    
    print("Ранжирование ответов по качеству:")
    print("-" * 60)
    
    for rank, (response, score) in enumerate(ranked_responses, 1):
        print(f"\n{rank}. Score: {score:.4f}")
        print(f"   Ответ: {response}")

if __name__ == "__main__":
    try:
        # Проверяем, что модель существует
        model_path = "./reward_model_output"
        if not os.path.exists(model_path):
            print(f"❌ Reward model не найдена в {model_path}")
            print("Сначала запустите: python train_reward_model.py")
            exit(1)
        
        # Запускаем демонстрации
        demo_basic_usage()
        demo_conversation_evaluation()
        demo_response_ranking()
        
        print("=== Демонстрация завершена ===")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
