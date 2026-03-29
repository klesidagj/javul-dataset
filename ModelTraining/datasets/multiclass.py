# ModelTraining/datasets/multiclass.py
from .base import CodeSnippetDataset
from ModelTraining.data.selectors import TopCWEMulticlassSelector


class MulticlassCodeSnippetDataset(CodeSnippetDataset):
    def __init__(self, cwe2id, *args, **kwargs):
        selector = TopCWEMulticlassSelector(cwe2id)
        super().__init__(*args, selector=selector, **kwargs)