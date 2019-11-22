from data import *
from typing import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
from model import Encoder, Decoder, Seq2Seq
from ot_dataset import OTDataset
import time
import math
import random
import argparse
from beam import beam_search


def train(model, iterator, optimizer, criterion, clip):

  model.train()

  epoch_loss = 0

  for i, batch in enumerate(iterator):

    src = batch.src
    trg = batch.trg

    optimizer.zero_grad()

    output = model(src, trg)

    # trg = [trg sent len, batch size]
    # output = [trg sent len, batch size, output dim]

    output = output[1:].view(-1, output.shape[-1])
    trg = trg[1:].view(-1)

    # trg = [(trg sent len - 1) * batch size]
    # output = [(trg sent len - 1) * batch size, output dim]

    loss = criterion(output, trg)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()

    epoch_loss += loss.item()

  return epoch_loss / len(iterator)


def evaluate(
        model,
        iterator,
        criterion,
        TRG,
        SRC,
        print_results=False,
        beam=False):

  model.eval()

  epoch_loss = 0

  with torch.no_grad():

    for i, batch in enumerate(iterator):

      src = batch.src
      trg = batch.trg

      output = model(src, trg, 0)  # turn off teacher forcing
      if beam:
        beam_search(output, TRG)

      # trg = [trg sent len, batch size]
      # output = [trg sent len, batch size, output dim]
      # grail = ['<unk>', '*Ident-IO(voi)', 'Agree', '*D', '*D_sigma', '<eos>']
      # for i in range(output.shape[1]):
      #   readable_output = list(map(lambda x: TRG.vocab.itos[x], torch.argmax(
      #       output[:, i, :], 1).tolist()))
      #   if readable_output == grail:
      #     print('found grail!')
      if print_results and i % random.randint(
              1, 20) or i % random.randint(1, 15) == 0:
        source = list(
            map(lambda x: SRC.vocab.itos[x], src[:, 0].tolist()))[1:-1]
        print(''.join(source).replace('<sep>', ', '), end=' & ')
        output_list = list(map(lambda x: TRG.vocab.itos[x], torch.argmax(
            output[:, 0, :], 1).tolist()))[1:-1]
        print(' >> '.join(output_list))

      output = output[1:].view(-1, output.shape[-1])
      trg = trg[1:].view(-1)

      # trg = [(trg sent len - 1) * batch size]
      # output = [(trg sent len - 1) * batch size, output dim]

      loss = criterion(output, trg)

      epoch_loss += loss.item()

  return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs


# TODO consider moving main part to seperate file
if __name__ == "__main__":
  from examples import end_voi_examples, hypo_voi_examples, star_agree_examples, star_agree_double_vowel_examples, star_agree_double_c_examples
  parser = argparse.ArgumentParser(
      description='Train and evaluate an OT constraint learner.')
  parser.add_argument('--unshuffle', '-u', action='store_true',
                      help='whether to not shuffle the entire dataset')
  parser.add_argument('--beam', '-b', action='store_true',
                      help='whether to not perform beam searches')
  parser.add_argument(
      '--epochs',
      '-e',
      action='store',
      type=int,
      default=5,
      help='number of epochs')
  parser.add_argument(
      '--min_word_examples',
      action='store',
      type=int,
      default=10,
      help='minimum number of examples per word')
  parser.add_argument(
      '--max_word_examples',
      action='store',
      type=int,
      default=100,
      help='maximum number of examples per word')
  parser.add_argument(
      '--min_pair_examples',
      action='store',
      type=int,
      default=10,
      help='minimum number of examples per word ranking pair')
  parser.add_argument(
      '--max_pair_examples',
      action='store',
      type=int,
      default=100,
      help='maximum number of examples per word ranking pair')
  parser.add_argument(
      '--test', '-t',
      action='store',
      type=str,
      default='all',
      help='what test set should be composed of')

  args = parser.parse_args()

  torch.cuda.empty_cache()

  SRC = Field()
  TRG = Field()
  def ot_dataset(examples): return OTDataset(examples, fields=(SRC, TRG))

  def gen_examples_with_params(words_and_rankings):
    return gen_all_examples(words_and_rankings,
                            args.min_pair_examples,
                            args.max_pair_examples,
                            args.min_word_examples,
                            args.max_word_examples)

  if args.test == 'all':
    words_and_rankings = end_voi_examples + hypo_voi_examples + star_agree_examples + \
        star_agree_double_c_examples + star_agree_double_vowel_examples
  elif args.test == 'double_vowel':
    words_and_rankings = end_voi_examples + hypo_voi_examples + \
        star_agree_examples + star_agree_double_c_examples
  else:
    raise(NotImplementedError(f'Test set {args.test} not implemented'))
  tupled_examples = gen_examples_with_params(words_and_rankings)
  # tupled_hypo_examples = gen_all_examples(hypo_voi_examples)
  if not args.unshuffle:
    print('shuffling')
    random.shuffle(tupled_examples)  # TODO make shuffling a param
  valid_split = int(len(tupled_examples) * 0.5)
  test_split = int(len(tupled_examples) * 0.75)
  train_data = OTDataset(
      tupled_examples[:valid_split], fields=(SRC, TRG))
  valid_data = OTDataset(
      tupled_examples[valid_split:test_split], fields=(SRC, TRG))
  if args.test == 'all':
    test_data = ot_dataset(tupled_examples[test_split:])
  elif args.test == 'double_vowel':
    test_data = ot_dataset(gen_examples_with_params(
        star_agree_double_vowel_examples))
  else:
    raise(NotImplementedError(f'Test set {args.test} not implemented'))

  print(f'train size: {len(train_data)}')
  print(f'validation size: {len(valid_data)}')
  print(f'test size: {len(test_data)}')

  SRC.build_vocab(train_data)
  TRG.build_vocab(train_data)
  print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
  print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

  BATCH_SIZE = 128
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
      (train_data, valid_data, test_data),
      batch_size=BATCH_SIZE,
      device=device)

  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  ENC_EMB_DIM = 256
  DEC_EMB_DIM = 256
  HID_DIM = 512
  N_LAYERS = 2
  ENC_DROPOUT = 0.5
  DEC_DROPOUT = 0.5

  enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
  dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

  model = Seq2Seq(enc, dec, device).to(device)

  def init_weights(m):
    for name, param in m.named_parameters():
      nn.init.uniform_(param.data, -0.08, 0.08)

  model.apply(init_weights)

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f'The model has {count_parameters(model):,} trainable parameters')

  optimizer = optim.Adam(model.parameters())

  PAD_IDX = TRG.vocab.stoi['<pad>']

  criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

  N_EPOCHS = args.epochs
  CLIP = 1

  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(
        model,
        valid_iterator,
        criterion,
        TRG,
        SRC, print_results=False)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), 'tut1-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(
        f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(
        f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

  model.load_state_dict(torch.load('tut1-model.pt'))

  test_loss = evaluate(
      model,
      test_iterator,
      criterion,
      TRG, SRC,
      print_results=True, beam=args.beam)

  print(
      f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

  torch.cuda.empty_cache()
