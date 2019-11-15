
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
from abc import ABC, abstractmethod


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


if __name__ == "__main__":
  # start English z suffix devoicing examples
  consonant = u_seg(['k', 'd', 'h', 't', 'ʔ', 'w'])
  vowel = u_seg(['æ', 'ɔ', 'ɛ', 'e', 'I'])
  s_z_seg = PossibleSegments([('s', 0.8), ('z', 0.2)])
  f_v_seg = PossibleSegments([('f', 0.8), ('v', 0.2)])

  cat_sz = [consonant, vowel, seg('t'), s_z_seg]
  gz = [consonant, vowel, seg('g'), seg('z')]
  nz = [consonant, vowel, seg('n'), seg('z')]
  five_f_v_th = [consonant, consonant, vowel, consonant, f_v_seg, seg('θ')]
  four_f_v_th = [consonant, vowel, consonant, f_v_seg, seg('θ')]
  two_vowel_t_theta = [consonant, vowel, vowel, seg('t'), seg('θ')]
  one_vowel_t_theta = [consonant, vowel, seg('t'), seg('θ')]
  n_theta = [consonant, vowel, seg('n'), seg('θ')]

  end_voi_words = list(map(Word, [
      cat_sz,
      gz,
      nz,
      five_f_v_th,
      four_f_v_th,
      two_vowel_t_theta,
      one_vowel_t_theta,
      n_theta]))

  agree, ident_voi, star_d, star_d_sigma = 'Agree', '*Ident-IO(voi)', '*D', '*D_sigma'
  english_voi: Ranking = [agree, ident_voi, star_d, star_d_sigma]
  non_english_voi: Ranking = [ident_voi, agree, star_d, star_d_sigma]
  voi_rankings = PossibleRankings([(english_voi, 0.8), (non_english_voi, 0.2)])

  end_voi_examples = [(word, voi_rankings) for word in end_voi_words]
  # end English z suffix devoicing examples

  words_and_rankings = end_voi_examples  # TODO will add more concantenated here

  train_data = gen_all_examples(words_and_rankings)
  # TODO make these splits instead of the same
  valid_data = gen_all_examples(words_and_rankings)
  test_data = gen_all_examples(words_and_rankings)
  SRC = Field()
  TRG = Field()

  SRC.build_vocab(train_data)
  TRG.build_vocab(train_data)
  print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")

  BATCH_SIZE = 128
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size=BATCH_SIZE,
      device=device)
