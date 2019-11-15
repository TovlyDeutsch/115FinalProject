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
import copy


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


def evaluate(model, iterator, criterion):

  model.eval()

  epoch_loss = 0

  with torch.no_grad():

    for i, batch in enumerate(iterator):

      src = batch.src
      trg = batch.trg

      output = model(src, trg, 0)  # turn off teacher forcing

      # trg = [trg sent len, batch size]
      # output = [trg sent len, batch size, output dim]

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


if __name__ == "__main__":
  from examples import end_voi_examples
  SRC = Field()
  TRG = Field()
  words_and_rankings = end_voi_examples  # TODO will add more concantenated here
  tupled_examples = gen_all_examples(words_and_rankings)
  train_data = OTDataset(
      tupled_examples, fields=(SRC, TRG))
  # TODO make these splits instead of the same
  valid_data = OTDataset(
      tupled_examples, fields=(SRC, TRG))
  test_data = OTDataset(
      tupled_examples, fields=(SRC, TRG))

  SRC.build_vocab(train_data)
  TRG.build_vocab(train_data)
  print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")

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

  N_EPOCHS = 10
  CLIP = 1

  best_valid_loss = float('inf')

  for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

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
