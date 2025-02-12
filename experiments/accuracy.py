import re
import string
from typing import Callable, List, Tuple
from collections import Counter
class Accuracy:
    def __init__(self):
        self._num_correct = 0
        self._total_score = 0
        self._num_total = 0
    def normalize_answer(self, s: str) -> str:
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction
    
    def get_score(self, ground_truth: str, prediction: str) -> float:
        score, _ = self.calculate_score(ground_truth, prediction)
        return score

    def update(self, predicted: str, target: str) -> None:
        is_correct = predicted == target
        score, _ = self.calculate_score(target, predicted)
        self._total_score += score
        self._num_correct += int(is_correct)
        self._num_total += 1

    def get(self) -> float:
        # return self._num_correct / self._num_total
        return self._total_score / self._num_total

    def print(self):
        accuracy = self.get()
        print(f"Accuracy: {accuracy*100:.1f}% "
              f"({self._num_correct}/{self._num_total})")