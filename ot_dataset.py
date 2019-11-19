from torchtext.datasets import TranslationDataset
import torchtext.data as data


class OTDataset(TranslationDataset):
  """Defines a dataset for translation between output examples and OT constraint rankings."""

  def __init__(self, tupled_examples, fields, **kwargs):
    """Create a TranslationDataset given paths and fields.
    Arguments:
        path: Common prefix of paths to the data files for both languages.
        exts: A tuple containing the extension to path for each language.
        fields: A tuple containing the fields that will be used for data
            in each language.
        Remaining keyword arguments: Passed to the constructor of
            data.Dataset.
    """
    if not isinstance(fields[0], (tuple, list)):
      fields = [('src', fields[0]), ('trg', fields[1])]

    examples = []
    for src_words, trg_ranking in tupled_examples:
      examples.append(data.Example.fromlist(
          [src_words, trg_ranking], fields))

    super(TranslationDataset, self).__init__(examples, fields, **kwargs)
