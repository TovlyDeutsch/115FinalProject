
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
  def __init__(self, choices: List[Tuple[str, float]]):
    self.possible_segments = [x[0] for x in choices]
    self.probs = [x[1] for x in choices]

  def sample(self) -> str:
    return np.random.choice(self.possible_segments, 1, p=self.probs)[0]


class Example():
  def __init__(self, src, trg):
    self.src = src
    self.trg = trg


def gen_seq(segments: Sequence[Sampleable]):
  """This generates multiple intances of a word/input seperated by <sep>"""
  src = ['<sos>']
  number_of_outputs = randint(5, 15)  # TODO remove magic numbers here
  for i in range(number_of_outputs):
    for seg in segments:
      src.append(seg.sample())
      src.append('<sep>')
  src[-1] = '<eos>'
  return src


def gen_example(segments: List[Sampleable], constraints: List[Sampleable]):
  example = Example(gen_seq(segments), gen_seq(constraints))
  return example


def gen_examples_for_input(
        word: List[Sampleable],
        ranking: List[Sampleable],
        min_num: int,
        max_num: int):
  # TODO maybe change from unifrom random to dome dist weight toward larger
  # nums
  num_examples = randint(min_num, max_num)
  return [gen_example(word, ranking) for i in range(num_examples)]


def gen_all_examples(
        words_and_rankings: List[Tuple[List[Sampleable], List[Sampleable]]]):
  examples = []
  min_number_of_examples = 10
  max_number_of_examples = 100
  for word, ranking in words_and_rankings:
    examples += gen_examples_for_input(word,
                                       ranking,
                                       min_number_of_examples,
                                       max_number_of_examples)
  return examples


def seg(char: str):
  return Sampleable([(char, 1.0)])


if __name__ == "__main__":
  # start English z suffix devoicing examples
  s_z_seg = Sampleable([('s', 0.8), ('z', 0.2)])
  cat_sz = [seg('k'), seg('æ'), seg('t'), s_z_seg]
  dogz = [seg('d'), seg('ɔ'), seg('g'), seg('z')]
  henz = [seg('h'), seg('ɛ'), seg('n'), seg('z')]
  f_v_seg = Sampleable([('f', 0.8), ('v', 0.2)])
  twel_f_v_th = [seg('t'), seg('w'), seg('ɛ'), seg('l'), f_v_seg, seg('θ')]
  eigth = [seg('ʔ'), seg('e'), seg('I'), seg('t'), seg('θ')]
  tenth = [seg('t'), seg('ɛ'), seg('n'), seg('θ')]
  end_voi_words = [cat_sz, dogz, henz, twel_f_v_th, eigth, tenth]

  agree, ident_voi, star_d, star_d_sigma = 'Agree', '*Ident-IO(voi)', '*D', '*D_sigma'
  english_voi = [agree, ident_voi, star_d, star_d_sigma]
  non_english_voi = [ident_voi, agree, star_d, star_d_sigma]
  voi_rankings = [(english_voi, 0.8), (non_english_voi, 0.2)]
  end_voi_examples = [(word, voi_rankings) for word in end_voi_words]
    # start English z suffix devoicing examples

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

  # segs = []
  # seg = Sampleable([(1, 0.8), (2, 0.2)])
  # for i in range(10):
  #   segs.append(seg.sample())
  # print(segs)
