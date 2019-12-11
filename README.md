# 115FinalProject

This repository contains the code used for the experiments for my final project in Linguistics 115.

To run the experiments, first install the requirements with 

```pip install -r requirements.txt```

Then, you can run some of the example commands in commands.txt. An example command is shown below.

```python train_and_eval.py --min_word_examples 1 --max_word_examples 5 --min_pair_examples 1 --max_pair_examples 10 -e 20```

`-e` refers to the number of epochs. `min_word_examples` and `min_word_examples` refer to the minimum and maximum number of concrete
surface forms. Similarly, `min_pair_examples` and `max_pair_examples` refer to the minimum and maximum number of concrete examples.
