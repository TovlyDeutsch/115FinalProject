
from scipy import stats
from typing import *
from functools import partial, reduce
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from functools import reduce
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator
from abc import ABC

T = TypeVar('T')


class Sampleable(ABC, Generic[T]):
  def __init__(self, choices: List[Tuple[T, float]]):
    self.possible_choices = [x[0] for x in choices]
    self.probs = [x[1] for x in choices]

  def sample(self) -> T:
    indicies = [x for x in range(len(self.possible_choices))]
    # index storage avoids np.random.choice's inability to work with 2d lists
    chosen_index = np.random.choice(indicies, 1, p=self.probs)[0]
    return self.possible_choices[chosen_index]


Constraint = str
Ranking = List[Constraint]


class Segment():
  def __init__(self, character: str):
    self.char = character
    from segments import phones, voiced, obstruents
    assert(character in phones)
    self.voiced = character in voiced
    self.obstruent = character in obstruents


class PossibleRankings(Sampleable):
  def __init__(self, choices: List[Tuple[Ranking, float]]):
    super().__init__(choices)


class PossibleSegments(Sampleable):
  def __init__(self, choices: List[Tuple[Segment, float]]):
    super().__init__(choices)

  def all_prop(self, prop: str) -> bool:
    return reduce(
        lambda acc, seg: acc and getattr(seg, prop),
        self.possible_choices,
        True)

  def all_voiced(self) -> bool:
    return self.all_prop('voiced')

  def all_obstruent(self) -> bool:
    return self.all_prop('obstruent')


class Word():
  def __init__(self, segments: List[PossibleSegments]):
    self.segments = segments

  def gen_sample_output(self) -> List[str]:
    """This generates multiple intances of a word/input seperated by <sep>"""
    src = ['<sos>']
    number_of_outputs = random.Random(0).randint(
        10, 20)  # TODO remove magic numbers here
    for i in range(number_of_outputs):
      for seg in self.segments:
        src.append(seg.sample().char)
      src.append('<sep>')
    src[-1] = '<eos>'
    return src


def gen_examples_for_word_and_rankings(
        word: Word,
        rankings: PossibleRankings,
        min_num: int,
        max_num: int) -> List[Tuple[List[str], Ranking]]:
  # TODO maybe change from uniform random to dome dist weight toward larger
  num_examples = random.Random(0).randint(min_num, max_num)
  return [(word.gen_sample_output(), rankings.sample())
          for i in range(num_examples)]


def gen_all_examples(
        words_and_rankings: List[Tuple[Word, PossibleRankings]]) -> List[Tuple[List[str], Ranking]]:
  examples = []
  min_number_of_examples = 20  # TODO maybe make this command line arg and/or param
  max_number_of_examples = 100
  for word, rankings in words_and_rankings:
    examples += gen_examples_for_word_and_rankings(word,
                                                   rankings,
                                                   min_number_of_examples,
                                                   max_number_of_examples)
  return examples


def seg(char: str) -> PossibleSegments:
  return PossibleSegments([(Segment(char), 1.0)])


def single_ranking(ranking: Ranking) -> PossibleRankings:
  return PossibleRankings([(ranking, 1.0)])


def u_seg(chars: Collection[str]) -> PossibleSegments:
  return PossibleSegments([(Segment(char), 1 / len(chars)) for char in chars])
