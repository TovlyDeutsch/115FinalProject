import tensorflow as tf
from itertools import groupby
import math
import numpy as np

# TODO make beam_k cmd line arg


def beam_search(output, TRG, beam_k=15):
  decodes, k_probabilities = tf.nn.ctc_beam_search_decoder(
      inputs=output.cpu().detach().numpy(), sequence_length=np.full(
          (output.shape[1]), output.shape[0]), top_paths=beam_k)
  top_k_decodes = []
  for k in range(min(beam_k, len(decodes))):
    decode = decodes[k].values.numpy()
    # prob = k_probabilities[k].numpy()
    char_list = []
    for i in range(len(decodes[k].values)):
      char_list.append(TRG.vocab.itos[decode[i] - 1])
    batch_seqs = [
        list(group) for k,
        group in groupby(
            char_list,
            lambda x: x == "<pad>" or x == '<eos>' or x == '<unk>') if not k]
    # print(k_probabilities[0, k].numpy())
    # print(k_probabilities[1, k].numpy())
    # first index is batch num and seond is k
    if math.exp(k_probabilities[0, 0].numpy()) - \
            math.exp(k_probabilities[0, 1].numpy()) > 0.1:
      print(k_probabilities[0, 0].numpy())
      print(k_probabilities[0, 1].numpy())
      print(batch_seqs[0])
      # print(batch_seqs[beam_k - 1])
