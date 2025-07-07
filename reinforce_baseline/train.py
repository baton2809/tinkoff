import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.device_utils import get_device, clear_memory_before
from reward_model.config.training_config import TrainingConfig
from reward_model.inference import RewardModelInference


class ReinforceTrainer:
    def __init__(self):
        self._setup_config()
        self._setup_training_state()
        self._initialize_models()

    def _setup_config(self) -> None:
        self.config = TrainingConfig()
        self.trained_model_path = self.config.output_dir
        self.reward_model_path = self.config.output_dir
        self.dataset_path = self.config.data_dir

        self.kl_coef = self.config.kl_coef
        self.baseline_momentum = self.config.baseline_momentum
        self.max_length = self.config.max_length
        self.temperature = self.config.temperature
        self.num_episodes = self.config.num_episodes
        self.reinforce_batch_size = self.config.reinforce_batch_size

    def _setup_training_state(self) -> None:
        self.baseline = None
        self.optimizer = None
        self.device = get_device()
        self.best_examples = []
        self.training_history = []
        self.reward_cache = {}

    def _initialize_models(self) -> None:
        self.reward_model = RewardModelInference(self.reward_model_path)
        self.policy_model = None
        self.ref_model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.load_models()
        self._load_datasets()

    def _load_datasets(self) -> None:
        self.train_dataset = self._load_dataset('train')
        self.val_dataset = self._load_dataset('validation')

    def _run_training_episode(self, episode: int, batch_size: int) -> dict:
        batch_metrics = {
            'loss': 0.0,
            'rewards': [],
            'kls': [],
            'advantages': []
        }

        self.policy_model.train()

        self.training_history.append({
            'episode': episode,
            'baseline': self.baseline
        })

        for batch_idx in range(batch_size):
            try:
                sample_idx = np.random.randint(0, len(self.train_dataset))
                sample = self.train_dataset[sample_idx]
                prompt = sample['prompt']
                generated_text, policy_log_prob = self.generate_text(prompt)

                if not isinstance(generated_text, str):
                    generated_text = self.tokenizer.decode(generated_text, skip_special_tokens=True)

                reward = self.get_reward(generated_text)
                kl_penalty = self.get_kl_penalty(prompt, generated_text, policy_log_prob)
                adjusted_reward = reward - self.kl_coef * kl_penalty.item()

                self.baseline = adjusted_reward if self.baseline == 0.0 else \
                    (self.baseline_momentum * self.baseline +
                     (1 - self.baseline_momentum) * adjusted_reward)

                advantage = adjusted_reward - self.baseline

                if policy_log_prob.requires_grad:
                    batch_metrics['loss'] += -advantage * policy_log_prob
                batch_metrics['rewards'].append(reward)
                batch_metrics['kls'].append(kl_penalty.item())
                batch_metrics['advantages'].append(advantage)

                if reward > self.baseline:
                    self.best_examples.append({
                        'prompt': prompt,
                        'text': generated_text,
                        'reward': reward
                    })

            except Exception as e:
                print(f"Ошибка в батче №{batch_idx}: {e}")
                continue

        return batch_metrics if batch_metrics['rewards'] else None

    @clear_memory_before
    def load_models(self) -> None:
        print("Загрузка предобученных моделей...")

        print(f"Загрузка токенизатора: {self.trained_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.trained_model_path)

        print(f"Загрузка SFT: {self.trained_model_path}")
        self.policy_model = AutoModelForCausalLM.from_pretrained(self.trained_model_path).to(self.device)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.trained_model_path).to(self.device)

        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.config.learning_rate)

        print(f"Загрузка Reward Model: {self.reward_model_path}")

    @clear_memory_before
    def generate_text(self, prompt: str) -> Tuple[str, torch.Tensor]:
        """
        Generate text from policy model with sampling.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[-1]

        outputs = self.policy_model.generate(
            **inputs,
            max_new_tokens=self.max_length,
            do_sample=True,
            temperature=self.temperature,
            top_k=0,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        generated_tokens = outputs.sequences[:, input_length:]

        if len(outputs.scores) == 0:
            return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), torch.tensor(0.0)

        log_probs_list = []
        for i, score in enumerate(outputs.scores):
            if i < generated_tokens.shape[1]:
                log_prob = torch.log_softmax(score, dim=-1)
                token_id = generated_tokens[0, i].unsqueeze(0)
                token_log_prob = torch.gather(log_prob, 1, token_id.unsqueeze(-1)).squeeze(-1)
                log_probs_list.append(token_log_prob)

        if log_probs_list:
            total_log_prob = torch.stack(log_probs_list).sum()
        else:
            total_log_prob = torch.tensor(0.0, device=self.device, requires_grad=True)

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return generated_text, total_log_prob

    @clear_memory_before
    def get_reward(self, text: str) -> float:
        """
        Get reward score for generated text.
        """
        if isinstance(text, torch.Tensor):
            text = self.tokenizer.decode(text, skip_special_tokens=True)

        text = text.strip()

        if hasattr(self, '_last_prompt') and self._last_prompt in text:
            text = text.replace(self._last_prompt, "").strip()

        if len(text) < 5:
            return -0.5

        try:
            if text in self.reward_cache:
                return self.reward_cache[text]

            reward = self.reward_model.score_text(text)
            reward = max(-10.0, min(10.0, reward))
            self.reward_cache[text] = reward
            return reward
        except Exception as e:
            print(f"Ошибка оценки Reward model: '{text[:50]}...': {e}")
            return -1.0

    @clear_memory_before
    def get_kl_penalty(self, prompt: str, generated_text: str, policy_log_prob: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence penalty between policy and reference model.
        """
        try:
            self._last_prompt = prompt

            if prompt in generated_text:
                response_only = generated_text.replace(prompt, "").strip()
            else:
                response_only = generated_text

            if len(response_only) == 0:
                return torch.tensor(0.0, device=self.device)

            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            response_inputs = self.tokenizer(response_only, return_tensors="pt").to(self.device)

            with torch.no_grad():
                full_text = prompt + response_only
                full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                ref_outputs = self.ref_model(**full_inputs)
                ref_logits = ref_outputs.logits

                prompt_length = prompt_inputs.input_ids.shape[-1]
                response_logits = ref_logits[:, prompt_length - 1:-1]

                ref_log_probs = torch.log_softmax(response_logits, dim=-1)

                response_token_ids = response_inputs.input_ids
                if response_token_ids.shape[-1] > 0:
                    min_length = min(ref_log_probs.shape[1], response_token_ids.shape[-1])
                    if min_length > 0:
                        ref_token_log_probs = torch.gather(
                            ref_log_probs[:, :min_length],
                            2,
                            response_token_ids[:, :min_length].unsqueeze(-1)
                        ).squeeze(-1)
                        ref_total_log_prob = ref_token_log_probs.sum()
                    else:
                        ref_total_log_prob = torch.tensor(0.0, device=self.device)
                else:
                    ref_total_log_prob = torch.tensor(0.0, device=self.device)

            kl_penalty = policy_log_prob - ref_total_log_prob
            kl_penalty = torch.clamp(kl_penalty, -10.0, 10.0)

            return kl_penalty

        except Exception as e:
            print(f"Ошибка KL: {e}")
            return torch.tensor(0.0, device=self.device)

    @clear_memory_before
    def train(self, num_episodes: int = 1000, batch_size: int = 4,
              patience: int = 50, val_freq: int = 10) -> None:
        self._setup_training_loop(num_episodes, batch_size)

        for episode in range(1, num_episodes + 1):
            batch_metrics = self._run_training_episode(episode, batch_size)

            if not batch_metrics:
                continue

            update_metrics = self._update_model(batch_metrics)

            if episode % 5 == 0:
                print(f"Episode {episode}: "
                      f"Reward={update_metrics.get('avg_reward', 0):.3f}, "
                      f"KL={update_metrics.get('avg_kl', 0):.3f}, "
                      f"Advantage={update_metrics.get('avg_advantage', 0):.3f}, "
                      f"Baseline={self.baseline:.3f}")

            if episode % val_freq == 0:
                self._run_validation(episode)
                if self.patience_counter >= patience:
                    print(f"Ранняя остановка на эпизоде {episode}")
                    break

        self._finalize_training()

    def _setup_training_loop(self, num_episodes: int, batch_size: int) -> None:
        print("\nОбучение REINFORCE с baseline")
        print(f"SFT модель: {self.trained_model_path}")
        print(f"Модель вознаграждения: {self.reward_model_path}")
        print(f"Обучение на {num_episodes} эпизодах с размером батча {batch_size}\n")

        self.baseline = 0.0
        self.best_reward = -float('inf')
        self.patience_counter = 0
        self.best_model = None

    def _load_dataset(self, split: str):
        try:
            dataset_path = Path(self.dataset_path) / split
            dataset = load_from_disk(str(dataset_path))
            print(f"Загружен {split} датасет: {len(dataset)} примеров")
            return dataset
        except Exception as e:
            print(f"Ошибка загрузки датасета {split}: {e}")
            return None

    def _update_model(self, batch_metrics: dict) -> dict:
        if not batch_metrics['rewards']:
            return {}

        avg_loss = batch_metrics['loss'] / len(batch_metrics['rewards'])
        avg_reward = np.mean(batch_metrics['rewards'])
        avg_kl = np.mean(batch_metrics['kls'])
        avg_advantage = np.mean(batch_metrics['advantages'])

        if isinstance(avg_loss, torch.Tensor) and avg_loss.requires_grad:
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return {
            'avg_reward': avg_reward,
            'avg_kl': avg_kl,
            'avg_advantage': avg_advantage,
            'avg_loss': avg_loss.item() if isinstance(avg_loss, torch.Tensor) else avg_loss
        }

    @clear_memory_before
    def _run_validation(self, episode: int) -> None:
        self.policy_model.eval()
        val_rewards = []

        with torch.no_grad():
            num_val_samples = min(10, len(self.val_dataset))
            for i in range(num_val_samples):
                try:
                    sample = self.val_dataset[i]
                    val_prompt = sample['prompt']
                    val_text, _ = self.generate_text(val_prompt)
                    val_rewards.append(self.get_reward(val_text))
                except Exception as e:
                    print(f"Ошибка валидации для примера {i}: {e}")
                    val_rewards.append(-1.0)

        avg_reward = np.mean(val_rewards) if val_rewards else -1.0

        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_model = {
                'episode': episode,
                'reward': avg_reward
            }
            self.patience_counter = 0
            print(f"Модель улучшена: Вознаграждение на валидации: {avg_reward:.4f}")
        else:
            self.patience_counter += 1

    def _finalize_training(self) -> None:
        print("\nОбучение завершено")
        print(f"Финальный baseline: {self.baseline:.4f}")

        if self.best_model:
            print(
                f"\nНаилучшее вознаграждение на валидации: {self.best_reward:.4f} на эпизоде {self.best_model['episode']}")
        else:
            print("Лучшая модель не найдена - возможно, обучение не удалось")


if __name__ == "__main__":
    trainer = ReinforceTrainer()
    test_samples = [trainer.val_dataset[i] for i in range(len(trainer.val_dataset))]

    pre_training_rewards = []
    print("\nОценка модели до обучения reinforce")
    for i, sample in enumerate(test_samples):
        try:
            if i % 100 == 0:
                prompt = sample['prompt']
                text, _ = trainer.generate_text(prompt)
                reward = trainer.get_reward(text)
                pre_training_rewards.append(reward)
                print(f"Пример {i + 1}, оценка: {reward:.4f}\n")
        except Exception as e:
            print(f"Ошибка при оценке примера {i + 1}: {e}")
            pre_training_rewards.append(-1.0)

    trainer.train(num_episodes=trainer.num_episodes, batch_size=trainer.reinforce_batch_size, patience=20, val_freq=10)

    print("\nОценка после обучения reinforce")
    post_training_rewards = []
    for i, sample in enumerate(test_samples):
        try:
            if i % 100 == 0:
                prompt = sample['prompt']
                text, _ = trainer.generate_text(prompt)
                reward = trainer.get_reward(text)
                post_training_rewards.append(reward)
                print(f"Пример {i + 1}, оценка: {reward:.4f}\n")
        except Exception as e:
            print(f"Ошибка при оценке примера {i + 1}: {e}")
            post_training_rewards.append(-1.0)

    avg_pre = np.mean(pre_training_rewards)
    avg_post = np.mean(post_training_rewards)
    improvement = avg_post - avg_pre

    print(f"\nРезультаты обучения:")
    print(f"Среднее вознаграждение до обучения: {avg_pre:.4f}")
    print(f"Среднее вознаграждение после обучения: {avg_post:.4f}")
    print(f"Улучшение: {improvement:.4f}")

    if improvement > 0:
        print("Обучение успешно - модель улучшилась")
    else:
        print("Модель не показала улучшения")
