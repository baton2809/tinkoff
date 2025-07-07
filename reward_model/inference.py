import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from reward_model.src.model_manager import RewardModelManager


class RewardModelInference:
    def __init__(self, model_path: str = "models/trained_model"):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to trained model directory
        """
        self.model_path = model_path
        self.model_manager = RewardModelManager()
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Модель не найдена {self.model_path}")

        print(f"Загрузка модели: {self.model_path}")
        self.model, self.tokenizer = self.model_manager.load_trained_model(self.model_path)
        print("Модель загружена")

    def score_text(self, text: str, max_length: int = 128) -> float:
        """
        Get reward score for text.
        
        Args:
            text: Input text to score
            max_length: Maximum sequence length
            
        Returns:
            Reward score
        """
        if self.model is None:
            self.load_model()

        return self.model_manager.get_reward_score(text, max_length)

    def compare_responses(self, prompt: str, response_a: str, response_b: str) -> dict:
        """
        Compare two responses for a prompt.
        
        Args:
            prompt: Input prompt
            response_a: First response
            response_b: Second response
            
        Returns:
            Comparison results
        """
        if self.model is None:
            self.load_model()

        return self.model_manager.compare_responses(prompt, response_a, response_b)

    def interactive_demo(self) -> None:
        if self.model is None:
            self.load_model()

        print("\nReward Model Interactive Demo")
        print("Commands:")
        print("  score <text> - Get reward score for text")
        print("  compare <prompt> | <response_a> | <response_b> - Compare responses")
        print("  quit - Exit")

        while True:
            try:
                user_input = input(">>> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if user_input.startswith('score '):
                    text = user_input[6:]
                    score = self.score_text(text)
                    print(f"Reward score: {score:.4f}")

                elif user_input.startswith('compare '):
                    parts = user_input[8:].split(' | ')
                    if len(parts) != 3:
                        print("Format: compare <prompt> | <response_a> | <response_b>")
                        continue

                    prompt, response_a, response_b = [p.strip() for p in parts]
                    result = self.compare_responses(prompt, response_a, response_b)

                    print(f"Response A score: {result['score_a']:.4f}")
                    print(f"Response B score: {result['score_b']:.4f}")
                    print(f"Difference: {result['difference']:.4f}")
                    print(f"Preferred: {result['preferred']}")

                else:
                    print("Unknown command. Use 'score <text>' or 'compare <prompt> | <response_a> | <response_b>'")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reward model inference")
    parser.add_argument("--model_path", default="models/trained_model",
                        help="Path to trained model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive demo")
    parser.add_argument("--text", help="Text to score")
    parser.add_argument("--prompt", help="Prompt for comparison")
    parser.add_argument("--response_a", help="First response for comparison")
    parser.add_argument("--response_b", help="Second response for comparison")

    args = parser.parse_args()

    inference = RewardModelInference(args.model_path)

    if args.interactive:
        inference.interactive_demo()
    elif args.text:
        score = inference.score_text(args.text)
        print(f"Reward score: {score:.4f}")
    elif args.prompt and args.response_a and args.response_b:
        result = inference.compare_responses(args.prompt, args.response_a, args.response_b)
        print(f"Response A score: {result['score_a']:.4f}")
        print(f"Response B score: {result['score_b']:.4f}")
        print(f"Difference: {result['difference']:.4f}")
        print(f"Preferred: {result['preferred']}")
    else:
        inference.load_model()

        prompt = "How to learn programming?"
        good_response = "Start with basics, practice regularly, build projects, and read documentation."
        bad_response = "Just Google it."

        result = inference.compare_responses(prompt, good_response, bad_response)

        print("Test Example:")
        print(f"Prompt: {prompt}")
        print(f"Good response: {good_response}")
        print(f"Bad response: {bad_response}\n")
        print(f"Good response score: {result['score_a']:.4f}")
        print(f"Bad response score: {result['score_b']:.4f}")
        print(f"Difference: {result['difference']:.4f}")
        print(f"Model prefers: {result['preferred']}\n")
