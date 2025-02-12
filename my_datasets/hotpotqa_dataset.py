import json
import glob
import os
import pandas as pd
import numpy as np
from typing import Union, List, Literal, Any, Dict
from abc import ABC

class HotpotQADataset(ABC):
    def __init__(
        self,
        split: Union[Literal['train'], Literal['dev'], Literal['test'], List[Literal['train'], Literal['dev'], Literal['test']]]
    ) -> None:
        """
        Initialize the HotpotQADataset.

        Args:
            split (str or list): The dataset split to load ('train', 'dev', 'test') or a list of splits.
        """
        if isinstance(split, list):
            data_paths = [f'datasets/HotpotQA/{item}.jsonl' for item in split]
            df = pd.DataFrame()
            for path in data_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Path {path} does not exist")
                else:
                    df = pd.concat([df, self._load_data(path)], axis=0, ignore_index=True)
            self._total_df = df
            print(f"Total number of questions across splits {split}: {len(self._total_df)}")
        else:
            self._split = split
            data_path = f"datasets/hotpotqa/{self._split}.json"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Path {data_path} does not exist")
            self._total_df: pd.DataFrame = self._load_data(data_path)
            print(f"Total number of questions in split '{self._split}': {len(self._total_df)}")

    @staticmethod
    def get_domain() -> str:
        return 'hotpotqa'

    @staticmethod
    def _load_data(data_path: str) -> pd.DataFrame:
        """
        Load HotpotQA data from a JSON file.

        Args:
            data_path (str): Path to the JSON file.

        Returns:
            pd.DataFrame: DataFrame containing the dataset.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = []
        for item in data:
            record = {
                '_id': item.get('_id', ''),
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'type': item.get('type', ''),
                'level': item.get('level', ''),
                'supporting_facts': item.get('supporting_facts', []),
                'context': item.get('context', [])
            }
            records.append(record)

        df = pd.DataFrame(records)
        print(f"Loaded {len(df)} records from {data_path}")
        return df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.Series), "Expected a pandas Series"
        return record

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        """
        Convert a record to the input format required by the model.

        Args:
            record (pd.Series): A single record from the dataset.

        Returns:
            Dict[str, Any]: A dictionary representing the model input.
        """
        question = record['question']
        contexts = record['context']  # List of [title, paragraphs]
        # Flatten the context into a single string
        context_text = "\n".join([f"{title}: {' '.join(paragraphs)}" for title, paragraphs in contexts])
        input_text = f"Question: {question}\nContext:\n{context_text}"
        input_dict = {"task": input_text}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """
        Postprocess the model's answer to match the expected format.

        Args:
            answer (str or list): The raw answer output by the model.

        Returns:
            str: The cleaned answer.
        """
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise ValueError("Expected answer to be a string or a list of strings.")
        answer = answer.strip()
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        """
        Extract the target answer from a record.

        Args:
            record (pd.Series): A single record from the dataset.

        Returns:
            str: The correct answer.
        """
        correct_answer = record['answer']
        if not isinstance(correct_answer, str):
            raise ValueError(f"Expected 'answer' to be a string, got {type(correct_answer)}")
        return correct_answer.strip()

    def remove(self, index: int) -> None:
        """
        Remove a record from the dataset by index.

        Args:
            index (int): The index of the record to remove.
        """
        self._total_df = self._total_df.drop(index).reset_index(drop=True)

    def reset_index(self) -> None:
        """
        Reset the DataFrame index.
        """
        self._total_df = self._total_df.reset_index(drop=True)

    # def filter_questions(self, filtered_questions: List[str]) -> None:
    #     """
    #     Filter the dataset to include only specified questions.

    #     Args:
    #         filtered_questions (List[str]): List of questions to retain.
    #     """
    #     # Assuming 'question' field contains the full question text
    #     initial_count = len(self._total_df)
    #     self._total_df = self._total_df[self._total_df['question'].isin(filtered_questions)].drop_duplicates(subset='question')
    #     self.reset_index()
    #     filtered_count = len(self._total_df)
    #     print(f"Filtered questions from {initial_count} to {filtered_count}")

    def save_to_csv(self, file_path: str) -> None:
        """
        Save the dataset to a CSV file.

        Args:
            file_path (str): Path to the output CSV file.
        """
        self._total_df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Dataset saved to {file_path}")

    def save_to_json(self, file_path: str) -> None:
        """
        Save the dataset to a JSON file.

        Args:
            file_path (str): Path to the output JSON file.
        """
        self._total_df.to_json(file_path, orient='records')
        print(f"Dataset saved to {file_path}")

    
