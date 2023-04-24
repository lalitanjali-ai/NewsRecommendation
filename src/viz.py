import importlib
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as utils
from dataset import DatasetTest, NewsDataset
from parameters import parse_args
from prepare_data import prepare_training_data
from preprocess import read_news, get_doc_input
from scipy.special import softmax


class ResultPredist:
    def __init__(self, args):
        rank = 0
        self.args = args

        self.article_file = os.path.join(args.test_data_dir, "news.tsv")
        self.articles = (
            pd.read_csv(self.article_file, sep="\t", header=None)
            .sort_values(by=[0])
            .set_index(0)
        )  # self.load_json(article_file)
        self.articles[3] = self.articles[3].astype(str)

        self.users = pd.read_csv(
            os.path.join(args.test_data_dir, f"behaviors_{rank}.tsv"),
            sep="\t",
            header=None,
        )
        self.users[3] = self.users[3].astype(str)
        self.users[4] = self.users[4].astype(str)

        self.results = test(args)

    def get_result(self, ix):

        history = self.users.iloc[ix, 3].split()
        article_hostory = [
            self.articles.loc[p, 3] for p in history if p in self.articles.index
        ]

        candid = self.users.iloc[ix, 4].split()
        candid_list = [x.split("-") for x in candid]
        cand_doc = [x[0] for x in candid_list]
        cand_doc_label = [int(x[1]) for x in candid_list]
        cand_doc = [
            self.articles.loc[p, 3] for p in cand_doc if p in self.articles.index
        ]

        result_dict = {
            "cand_doc": cand_doc,  # candidate articles
            "cand_label": cand_doc_label,  # candidate articles's click ground truth
            "hostory": article_hostory,  # user history click
            "auc": self.results["auc"][ix],
            "mrr": self.results["mrr"][ix],
            "ndcg5": self.results["ndcg5"][ix],
            "ndcg10": self.results["ndcg10"][ix],
            "score": softmax(
                self.results["scores"][ix]
            ),  # probability of click on each candidate
        }

        return result_dict

    # TODO: add titles for user that has clicked


def test(args):

    is_distributed = False
    rank = 0

    if args.enable_gpu:
        torch.cuda.set_device(rank)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)

    assert ckpt_path is not None, "No checkpoint found."
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    subcategory_dict = checkpoint["subcategory_dict"]
    category_dict = checkpoint["category_dict"]
    word_dict = checkpoint["word_dict"]

    dummy_embedding_matrix = np.zeros((len(word_dict) + 1, args.word_embedding_dim))
    module = importlib.import_module(f"model.{args.model}")
    model = module.Model(
        args, dummy_embedding_matrix, len(category_dict), len(subcategory_dict)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_gpu:
        model.cuda(rank)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news(
        os.path.join(args.test_data_dir, "news.tsv"), args, mode="test"
    )
    news_title, news_category, news_subcategory = get_doc_input(
        news, news_index, category_dict, subcategory_dict, word_dict, args
    )
    news_combined = np.concatenate(
        [x for x in [news_title, news_category, news_subcategory] if x is not None],
        axis=-1,
    )

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(
        news_dataset, batch_size=args.batch_size, num_workers=4
    )

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
                input_ids = input_ids.cuda(rank)
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)
    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    if rank == 0:
        doc_sim = 0
        for _ in tqdm(range(1000000)):
            i = random.randrange(1, len(news_scoring))
            j = random.randrange(1, len(news_scoring))
            if i != j:
                doc_sim += np.dot(news_scoring[i], news_scoring[j]) / (
                    np.linalg.norm(news_scoring[i]) * np.linalg.norm(news_scoring[j])
                )
        logging.info(f"News doc-sim: {doc_sim / 1000000}")

    data_file_path = os.path.join(args.test_data_dir, f"behaviors_{rank}.tsv")

    data_file_path = os.path.join(args.test_data_dir, f"behaviors_{rank}.tsv")

    def collate_fn(tuple_list):
        log_vecs = torch.FloatTensor([x[0] for x in tuple_list])
        log_mask = torch.FloatTensor([x[1] for x in tuple_list])
        news_vecs = [x[2] for x in tuple_list]
        labels = [x[3] for x in tuple_list]
        return (log_vecs, log_mask, news_vecs, labels)

    dataset = DatasetTest(data_file_path, news_index, news_scoring, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    from metrics import roc_auc_score, ndcg_score, mrr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    scores = []

    def print_metrics(rank, cnt, x):
        logging.info(
            "[{}] {} samples: {}".format(
                rank, cnt, "\t".join(["{:0.2f}".format(i * 100) for i in x])
            )
        )

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    def get_sum(arr):
        return [np.array(i).sum() for i in arr]

    local_sample_num = 0

    user = 0

    for cnt, (log_vecs, log_mask, news_vecs, labels) in enumerate(dataloader):
        local_sample_num += log_vecs.shape[0]

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(rank, non_blocking=True)
            log_mask = log_mask.cuda(rank, non_blocking=True)

        user_vecs = (
            model.user_encoder(log_vecs, log_mask)
            .to(torch.device("cpu"))
            .detach()
            .numpy()
        )

        for user_vec, news_vec, label in zip(user_vecs, news_vecs, labels):
            if label.mean() == 0 or label.mean() == 1:
                AUC.append(np.nan)
                MRR.append(np.nan)
                nDCG5.append(np.nan)
                nDCG10.append(np.nan)
                scores.append(np.nan)
                continue

            score = np.dot(news_vec, user_vec)

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)
            scores.append(score)

        if cnt % args.log_steps == 0:
            print_metrics(rank, local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))

    logging.info("[{}] local_sample_num: {}".format(rank, local_sample_num))
    print_metrics("*", local_sample_num, get_mean([AUC, MRR, nDCG5, nDCG10]))
    return {"auc": AUC, "mrr": MRR, "ndcg5": nDCG5, "ndcg10": nDCG10, "scores": scores}


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    utils.dump_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args.model_dir = "model"
    args.model = "NRMS"
    args.mode = "test"
    args.enable_gpu = False
    args.load_ckpt_name = "epoch-1.pt"
    if args.prepare:
        logging.info("Preparing training data...")
        total_sample_num = prepare_training_data(
            args.train_data_dir, args.nGPU, args.npratio, args.seed
        )

    results = ResultPredist(args)
    results.get_result(73000)
