# ModelTraining/datasets/binary.py
from .base import CodeSnippetDataset
from ModelTraining.data.selectors import BinaryVulnerabilitySelector


class BinaryCodeSnippetDataset(CodeSnippetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, selector=BinaryVulnerabilitySelector(), **kwargs)