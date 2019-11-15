
from scipy import stats
from typing import *
from functools import partial, reduce
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import numpy as np

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time
from abc import ABC


class Sampleable(ABC):
  def __init__(self, choices: List[Tuple[Any, float]]):
    self.possible_segments = [x[0] for x in choices]
    self.probs = [x[1] for x in choices]

  def sample(self) -> str:
    return np.random.choice(self.possible_segments, 1, p=self.probs)[0]


Constraint = str
Ranking = List[Constraint]


class PossibleRankings(Sampleable):
  def __init__(self, choices: List[Tuple[Ranking, float]]):
    super().__init__(choices)


class PossibleSegments(Sampleable):
  def __init__(self, choices: List[Tuple[str, float]]):
    super().__init__(choices)


class Word():
  def __init__(self, segments: List[PossibleSegments]):
    self.segments = segments

  def gen_sample_output(self):
    """This generates multiple intances of a word/input seperated by <sep>"""
    src = ['<sos>']
    number_of_outputs = randint(5, 15)  # TODO remove magic numbers here
    for i in range(number_of_outputs):
      for seg in self.segments:
        src.append(seg.sample())
        src.append('<sep>')
    src[-1] = '<eos>'
    return src


class Example():
  def __init__(self, src, trg):
    self.src = src
    self.trg = trg


def gen_examples_for_word_and_rankings(
        word: Word,
        rankings: PossibleRankings,
        min_num: int,
        max_num: int):
  # TODO maybe change from uniform random to dome dist weight toward larger
  num_examples = randint(min_num, max_num)
  return [Example(word.gen_sample_output(), rankings.sample())
          for i in range(num_examples)]


def gen_all_examples(
        words_and_rankings: List[Tuple[Word, PossibleRankings]]):
  examples = []
  min_number_of_examples = 20  # TODO maybe make this command line arg
  max_number_of_examples = 100
  for word, rankings in words_and_rankings:
    examples += gen_examples_for_word_and_rankings(word,
                                                   rankings,
                                                   min_number_of_examples,
                                                   max_number_of_examples)
  return examples


def seg(char: str):
  return PossibleSegments([(char, 1.0)])


def u_seg(chars: List[str]):
  return PossibleSegments([(char, 1 / len(chars)) for char in chars])
