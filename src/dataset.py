
from torch.utils.data import IterableDataset, Dataset
import numpy as np
import random

class DatasetTrain(IterableDataset):
    def __init__(self, filename, news_index, news_combined, args):
        super(DatasetTrain, self).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_combined = news_combined
        self.args = args

    def trans_to_nindex(self, nids):
        return [
            (self.news_index[i], False) if i in self.news_index else (0, True)
            for i in nids
        ]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if len(x) < fix_length:
            if padding_front:
                pad_x = [padding_value] * (fix_length - len(x)) + x
                mask = [0] * (fix_length - len(x)) + [1] * len(x)
            else:
                pad_x = x + [padding_value] * (fix_length - len(x))
                mask = [1] * len(x) + [0] * (fix_length - len(x))
        else:
            if padding_front:
                pad_x = x[-fix_length:]
                mask = [1] * fix_length
            else:
                pad_x = x[:fix_length]
                mask = [1] * fix_length

        return pad_x, np.array(mask, dtype="float32")

    def line_mapper(self, line):
        line = line.strip().split("\t")

        click_docs = line[3].split()
        sess_pos = line[4].split()
        sess_neg = line[5].split()

        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(line[3].split()), self.args.user_log_length
        )

        # Handle padding value
        user_feature = np.array(
            [
                self.news_combined[index]
                if not is_padding
                else np.zeros(self.news_combined.shape[1], dtype="int32")
                for index, is_padding in zip(click_docs, log_mask)
            ]
        )

        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.args.npratio)
        sample_news = neg[:label] + pos + neg[label:]
        # Handle padding value
        news_feature = np.array(
            [
                self.news_combined[index]
                if not is_padding
                else np.zeros(self.news_combined.shape[1], dtype="int32")
                for index, is_padding in sample_news
            ]
        )

        # # Updated to include abstract information
        return (
            user_feature[:, : self.args.num_words_title],
            user_feature[
            :,
            self.args.num_words_title : self.args.num_words_title
                                        + self.args.num_words_abstract,
            ],
            log_mask,
            news_feature[:, : self.args.num_words_title],
            news_feature[
            :,
            self.args.num_words_title : self.args.num_words_title
                                        + self.args.num_words_abstract,
            ],
            label,
        )

    def __iter__(self):
        file_iter = open(self.filename)
        # Use filter to remove None values returned by the line_mapper
        return filter(None, map(self.line_mapper, file_iter))

class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class DatasetTest(DatasetTrain):
    def __init__(self, filename, news_index, news_scoring, args):
        super(DatasetTrain).__init__()
        self.filename = filename
        self.news_index = news_index
        self.news_scoring = news_scoring
        self.args = args

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def line_mapper(self, line):

        line = line.strip().split("\t")
        click_docs = line[3].split()
        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_docs), self.args.user_log_length
        )
        user_feature = self.news_scoring[click_docs]

        candidate_news = self.trans_to_nindex(
            [i.split("-")[0] for i in line[4].split()]
        )
        labels = np.array([int(i.split("-")[1]) for i in line[4].split()])
        news_feature = self.news_scoring[candidate_news]

        return user_feature, log_mask, news_feature, labels

    def __iter__(self):
        file_iter = open(self.filename)
        return map(self.line_mapper, file_iter)