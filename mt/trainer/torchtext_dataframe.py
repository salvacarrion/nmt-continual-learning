from torchtext import data
from torchtext.legacy.datasets import TranslationDataset
import os


class CustomDataset(TranslationDataset):

    urls = []
    name = 'customdataset'
    dirname = ''

    @classmethod
    def splits(cls, exts, fields, root='.data', train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of the Multi30k dataset.

        Args:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """

        # TODO: This is a _HORRIBLE_ patch related to #208
        # 'path' can be passed as a kwarg to the translation dataset constructor
        # or has to be set (so the download wouldn't be duplicated). A good idea
        # seems to rename the existence check variable from path to something else
        if 'path' not in kwargs:
            expected_folder = os.path.join(root, cls.name)
            path = expected_folder if os.path.exists(expected_folder) else None
        else:
            path = kwargs['path']
            del kwargs['path']

        return super(CustomDataset, cls).splits(
            exts, fields, path, root, train, validation, test, **kwargs)

#
# class DataFrameDataset(data.Dataset):
#
#     def __init__(self, df, text_field, label_field, is_test=False, **kwargs):
#         fields = [('text', text_field), ('label', label_field)]
#         examples = []
#         for i, row in df.iterrows():
#             label = row.sentiment if not is_test else None
#             text = row.text
#             examples.append(data.Example.fromlist([text, label], fields))
#
#         super().__init__(examples, fields, **kwargs)
#
#     @staticmethod
#     def sort_key(ex):
#         return len(ex.text)
#
#     @classmethod
#     def splits(cls, text_field, label_field, train_df, val_df=None, test_df=None, **kwargs):
#         train_data, val_data, test_data = (None, None, None)
#
#         if train_df is not None:
#             train_data = cls(train_df.copy(), text_field, label_field, **kwargs)
#         if val_df is not None:
#             val_data = cls(val_df.copy(), text_field, label_field, **kwargs)
#         if test_df is not None:
#             test_data = cls(test_df.copy(), text_field, label_field, True, **kwargs)
#
#         return tuple(d for d in (train_data, val_data, test_data) if d is not None)
#
