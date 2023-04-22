from collections import Counter

from nltk.corpus import stopwords
from tqdm import tqdm
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import re

nltk.download('stopwords')
# Define a list of stopwords
stop_words = set(stopwords.words('english'))

def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


def preprocess_text(text):
    # Remove numbers and punctuation
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert to lowercase and tokenize
    words = word_tokenize(text.lower())

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join the words back into a string
    processed_text = ' '.join(words)

    return processed_text


def read_news(news_path, args, mode='train', bert_topics_train=None):
    news = {}
    abstract = {}
    category_dict = {}
    subcategory_dict = {}
    news_index = {}
    word_cnt = Counter()
    abstract_word_cnt = Counter()

    if args.use_topics and bert_topics_train is not None:
        bert_topics_train = bert_topics_train.set_index('News ID')
        bert_topics_index_dict = bert_topics_train.to_dict(orient='index')
        news_id_topic_dict = {k: v['topic_name'] for k, v in bert_topics_index_dict.items()}

    with open(news_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title1, abstract, url, _, _ = splited
            update_dict(news_index, doc_id)

            title = title1.lower()
            title = word_tokenize(title)

            abstract = abstract.lower()
            abstract = preprocess_text(abstract)
            abstract = word_tokenize(abstract)

            if args.use_topics == True:
                news_id = doc_id
                if news_id in news_id_topic_dict.keys():
                    topic_string = news_id_topic_dict[str(news_id)]
                    # topic=topic_string.split(',')
                    if isinstance(topic_string, (str, bytes)):
                        topic = preprocess_text(topic_string)
                        topic = word_tokenize(topic)
                        title = title + topic
                        title = list(set(title))

            update_dict(news, doc_id, [title, category, subcategory, abstract])

            if mode == 'train':
                if args.use_category:
                    update_dict(category_dict, category)
                if args.use_subcategory:
                    update_dict(subcategory_dict, subcategory)
                word_cnt.update(title)
                abstract_word_cnt.update(abstract)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > args.filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}

        abs_word = [k for k, v in abstract_word_cnt.items() if v > 100]
        abs_word_dict = {k: v for k, v in zip(abs_word, range(1, len(abs_word) + 1))}

        return news, news_index, category_dict, subcategory_dict, word_dict, abs_word_dict

    elif mode == 'test':
        return news, news_index

    else:
        assert False, 'Wrong mode!'


def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, abs_word_dict, args):

    news_num = len(news) + 1
    news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
    news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
    news_subcategory = np.zeros((news_num, 1), dtype='int32') if args.use_subcategory else None
    news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')

    for key in tqdm(news):

        title, category, subcategory, abstract = news[key]
        doc_index = news_index[key]

        for word_id in range(min(args.num_words_title, len(title))):
            if title[word_id] in word_dict:
                news_title[doc_index, word_id] = word_dict[title[word_id]]

        for word_id in range(min(args.num_words_abstract, len(abstract))):
            if abstract[word_id] in abs_word_dict:
                news_abstract[doc_index, word_id] = abs_word_dict[abstract[word_id]]

        if args.use_category:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        if args.use_subcategory:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

    return news_title, news_category, news_subcategory, news_abstract

# def read_news(news_path, args, mode='train'):
#     news = {}
#     category_dict = {}
#     subcategory_dict = {}
#     news_index = {}
#     word_cnt = Counter()
#
#     with open(news_path, 'r', encoding='utf-8') as f:
#         for line in tqdm(f):
#             splited = line.strip('\n').split('\t')
#             doc_id, category, subcategory, title, abstract, url, _, _ = splited
#             update_dict(news_index, doc_id)
#
#             title = title.lower()
#             title = word_tokenize(title)
#             update_dict(news, doc_id, [title, category, subcategory])
#             if mode == 'train':
#                 if args.use_category:
#                     update_dict(category_dict, category)
#                 if args.use_subcategory:
#                     update_dict(subcategory_dict, subcategory)
#                 word_cnt.update(title)
#
#     if mode == 'train':
#         word = [k for k, v in word_cnt.items() if v > args.filter_num]
#         word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
#         return news, news_index, category_dict, subcategory_dict, word_dict
#     elif mode == 'test':
#         return news, news_index
#     else:
#         assert False, 'Wrong mode!'
#
#
# def get_doc_input(news, news_index, category_dict, subcategory_dict, word_dict, args):
#     news_num = len(news) + 1
#     news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
#     news_category = np.zeros((news_num, 1), dtype='int32') if args.use_category else None
#     news_subcategory = np.zeros((news_num, 1), dtype='int32') if args.use_subcategory else None
#
#     for key in tqdm(news):
#         title, category, subcategory = news[key]
#         doc_index = news_index[key]
#
#         for word_id in range(min(args.num_words_title, len(title))):
#             if title[word_id] in word_dict:
#                 news_title[doc_index, word_id] = word_dict[title[word_id]]
#
#         if args.use_category:
#             news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
#         if args.use_subcategory:
#             news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
#
#     return news_title, news_category, news_subcategory
