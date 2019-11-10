
from scipy import stats
from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

class Sampleable():
    def __init__(self, choices: List[Tuple[int, float]]):
      possible_segments = [x[0] for x in choices]
      probs = [x[1] for x in choices]
      self.dist = stats.rv_discrete(name='custm', values=(possible_segments, probs))
    
    @abstractmethod
    def sample(self) -> str:
      pass

class Segment(Sampleable):
  def sample(self):
    phone_lookup = {
      0: '<sep>', 1: 's', 2: 'z', 3: 'd', 4: 't', 4: 'n', 5: 'a'
    }
    return phone_lookup[self.dist.rvs()]

class Constraint(Sampleable):
  def sample(self):
    con_lookup = {
      0: '<sep>', 1: 'agree', 2: '*D'
    }
    return con_lookup[self.dist.rvs()]

# class Sequence():
#   def __init__(self):
#     pass

class Example():
  def __init__(self, src, trg):
    self.src = src
    self.trg = trg

def gen_seq(segments: Sequence[Sampleable]):
  src = ['<sos>']
  number_of_outputs = randint(5, 15)
  for i in range(number_of_outputs):
    for seg in segments:
      src.append(seg.sample())
    if i != number_of_outputs - 1:
      src.append('<sep>')
  src.append('<eos>')
  return src

def gen_example(segments: List[Segment], constraints: List[Constraint]):
  example = Example(gen_seq(segments), gen_seq(constraints))
  return example


if __name__ == "__main__":
  s_z_seg = Segment([(1, 0.8), (2, 0.2)])
  test_word = [s_z_seg, s_z_seg, s_z_seg, s_z_seg]

  test_constraint = Constraint([(1, 0.8), (2, 0.2)])
  test_ranking = [test_constraint, test_constraint]

  train_data = [gen_example(test_word, test_ranking) for i in range(10)]
  valid_data = [gen_example(test_word, test_ranking) for i in range(10)]
  test_data = [gen_example(test_word, test_ranking) for i in range(10)]
  SRC = Field()
  TRG = Field()

  SRC.build_vocab(train_data)
  TRG.build_vocab(train_data)
  print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")

  BATCH_SIZE = 128
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
      (train_data, valid_data, test_data), 
      batch_size = BATCH_SIZE, 
      device = device)

    # segs = []
    # seg = Segment([(1, 0.8), (2, 0.2)])
    # for i in range(10):
    #   segs.append(seg.sample())
    # print(segs)