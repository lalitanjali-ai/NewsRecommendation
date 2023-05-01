import importlib
import logging
import os
import random
import subprocess
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from nltk.corpus import stopwords
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm

import utils
from dataset import DatasetTrain, DatasetTest, NewsDataset
from parameters import parse_args
from prepare_data import prepare_training_data, prepare_testing_data
from preprocess import read_news, get_doc_input
from metrics import roc_auc_score, ndcg_score, mrr_score
from torch.nn.utils.rnn import pad_sequence

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


def train(rank, args):
    if rank is None:
        is_distributed = False
        rank = 0
    else:
        is_distributed = True

    if is_distributed:
        utils.setuplogger()
        dist.init_process_group('nccl', world_size=args.nGPU, init_method='env://', rank=rank)

    if args.enable_gpu:
        torch.cuda.set_device(rank)

    if args.use_topics:
        bert_topics_train = pd.read_csv("bert_topics/train_small_bert_topics.tsv", sep='\t')

    news, news_index, category_dict, subcategory_dict, word_dict, abs_word_dict = read_news(
        os.path.join(args.train_data_dir, 'news.tsv'), args, mode='train', bert_topics_train=bert_topics_train)

    news_title, news_category, news_subcategory, news_abstract = get_doc_input(
        news, news_index, category_dict, subcategory_dict, word_dict, abs_word_dict, args)

    if args.use_category or args.use_subcategory:
        news_combined = np.concatenate(
            [x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)
        if args.use_category:
            args.num_words_title = args.num_words_title + 1
        if args.use_subcategory:
            args.num_words_title = args.num_words_title + 1
    elif args.model == 'NRMS_abstract':
        news_combined = np.concatenate([x for x in [news_title, news_abstract] if x is not None], axis=-1)
    else:
        news_combined = np.concatenate([x for x in [news_title] if x is not None], axis=-1)

    if args.word_embedding_type == 'bert':
        args.word_embedding_dim == 768

    if rank == 0:
        logging.info('Initializing word embedding matrix...')

    if args.word_embedding_type == 'bert':
        args.word_embedding_dim == 768
        embedding_matrix, have_word = utils.load_matrix_bert(word_dict, args.word_embedding_dim)
    else:
        embedding_matrix, have_word = utils.load_matrix(args.glove_embedding_path,
                                                        word_dict,
                                                        args.word_embedding_dim)
    if rank == 0:
        logging.info(f'Word dict length: {len(word_dict)}')
        logging.info(f'Have words: {len(have_word)}')
        logging.info(f'Missing rate: {(len(word_dict) - len(have_word)) / len(word_dict)}')

    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, embedding_matrix, len(category_dict), len(subcategory_dict))

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.enable_gpu:
        model = model.cuda(rank)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        print(model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{rank}.tsv')

    dataset = DatasetTrain(data_file_path, news_index, news_combined, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    logging.info('Training...')
    for ep in range(args.start_epoch, args.epochs):
        loss = 0.0
        accuary = 0.0

        for cnt, (user_titles, user_abstracts, log_mask, news_titles, news_abstracts, targets) in enumerate(dataloader):
            if args.enable_gpu:
                user_titles = user_titles.cuda(rank, non_blocking=True)
                user_abstracts = user_abstracts.cuda(rank, non_blocking=True)
                log_mask = log_mask.cuda(rank, non_blocking=True)
                news_titles = news_titles.cuda(rank, non_blocking=True)
                news_abstracts = news_abstracts.cuda(rank, non_blocking=True)
                targets = targets.cuda(rank, non_blocking=True)

            bz_loss, y_hat = model(user_titles, user_abstracts, log_mask, news_titles, news_abstracts, targets)
            loss += bz_loss.data.float()
            accuary += utils.acc(targets, y_hat)
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            # if cnt % args.log_steps == 0:
            #     logging.info(
            #         '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
            #             rank, cnt * args.batch_size, loss.data / cnt, accuary / cnt)
            #     )
            #
            # if rank == 0 and cnt != 0 and cnt % args.save_steps == 0:
            #     ckpt_path = os.path.join(args.model_dir, f'epoch-{ep + 1}-{cnt}.pt')
            #     torch.save(
            #         {
            #             'model_state_dict':
            #                 {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items()}
            #                 if is_distributed else model.state_dict(),
            #             'category_dict': category_dict,
            #             'word_dict': word_dict,
            #             'subcategory_dict': subcategory_dict,
            #             'abs_word_dict': abs_word_dict,
            #             'embedding_matrix': embedding_matrix,
            #
            #         }, ckpt_path)
            #     logging.info(f"Model saved to {ckpt_path}.")

        logging.info('Training finish.')

        if rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep + 1}.pt')
            torch.save(
                {
                    'model_state_dict':
                        {'.'.join(k.split('.')[1:]): v for k, v in model.state_dict().items()}
                        if is_distributed else model.state_dict(),
                    'category_dict': category_dict,
                    'subcategory_dict': subcategory_dict,
                    'word_dict': word_dict,
                    'abs_word_dict': abs_word_dict,
                    # 'embedding_matrix': embedding_matrix,

                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}.")


def test(rank, args):
    if rank is None:
        is_distributed = False
        rank = 0
    else:
        is_distributed = True

    if is_distributed:
        utils.setuplogger()
        dist.init_process_group('nccl', world_size=args.nGPU, init_method='env://', rank=rank)


    if args.use_topics:
        bert_topics_test = pd.read_csv("bert_topics/val_small_bert_topics.tsv", sep='\t')

    # Load model from checkpoint
    ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    abs_word_dict = checkpoint['abs_word_dict']

    news, news_index  = read_news(
        os.path.join(args.test_data_dir, 'news.tsv'), args, mode='test', bert_topics_train=bert_topics_test)

    news_title, news_category, news_subcategory, news_abstract = get_doc_input(
        news, news_index, category_dict, subcategory_dict, word_dict, abs_word_dict, args)

    if args.use_category or args.use_subcategory:
        news_combined = np.concatenate(
            [x for x in [news_title, news_category, news_subcategory] if x is not None], axis=-1)
        if args.use_category:
            args.num_words_title = args.num_words_title + 1
        if args.use_subcategory:
            args.num_words_title = args.num_words_title + 1
    elif args.model == 'NRMS_abstract':
        news_combined = np.concatenate([x for x in [news_title, news_abstract] if x is not None], axis=-1)
    else:
        news_combined = np.concatenate([x for x in [news_title] if x is not None], axis=-1)



    # embedding_matrix = checkpoint['embedding_matrix']
    dummy_embedding_matrix = np.zeros((len(word_dict) + 1, args.word_embedding_dim))

    module = importlib.import_module(f'model.{args.model}')
    model = module.Model(args, dummy_embedding_matrix, len(category_dict), len(subcategory_dict))
    model.load_state_dict(checkpoint['model_state_dict'])

    if args.enable_gpu:
        model = model.cuda(rank)

    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])


    model.eval()

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=4)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
                input_ids = input_ids.cuda(rank)
            b_size, d_size = input_ids.shape
            news_vec = model.news_encoder(input_ids[:b_size, :d_size//2], input_ids[:b_size, d_size//2:])
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)


    def custom_collate_fn(batch):
        user_titles, user_abstracts, log_mask, news_titles, news_abstracts, targets = zip(*batch)

        user_titles = pad_sequence([torch.tensor(t) for t in user_titles], batch_first=True, padding_value=0)
        user_abstracts = pad_sequence([torch.tensor(a) for a in user_abstracts], batch_first=True, padding_value=0)
        log_mask = torch.stack(
            [torch.tensor(mask) for mask in log_mask])  # Convert log_mask to a Tensor before stacking
        news_titles = pad_sequence([torch.tensor(t) for t in news_titles], batch_first=True, padding_value=0)
        news_abstracts = pad_sequence([torch.tensor(a) for a in news_abstracts], batch_first=True, padding_value=0)

        # Pad the targets to have the same length
        targets = pad_sequence([torch.tensor(target) for target in targets], batch_first=True, padding_value=0)

        logging.info("user_titles.shape", user_titles.shape)
        logging.info("user_abstracts.shape", user_abstracts.shape)
        logging.info("log_mask.shape", log_mask.shape)
        logging.info("news_titles.shape", news_titles.shape)
        logging.info("news_abstracts.shape", news_abstracts.shape)
        logging.info("targets.shape", targets.shape)

        return user_titles, user_abstracts, log_mask, news_titles, news_abstracts, targets

    data_file_path = os.path.join(args.test_data_dir, f"behaviors_{rank}.tsv")

    dataset = DatasetTest(data_file_path, news_index, news_scoring, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, collate_fn=custom_collate_fn)

    logging.info('Testing...')
    loss = 0.0
    accuracy = 0.0
    cnt = 0

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    with torch.no_grad():
        for user_titles, user_abstracts, log_mask, news_titles, news_abstracts, targets in dataloader:
            if args.enable_gpu:
                user_titles = user_titles.cuda(rank, non_blocking=True)
                user_abstracts = user_abstracts.cuda(rank, non_blocking=True)
                log_mask = log_mask.cuda(rank, non_blocking=True)
                news_titles = news_titles.cuda(rank, non_blocking=True)
                news_abstracts = news_abstracts.cuda(rank, non_blocking=True)
                targets = targets.cuda(rank, non_blocking=True)

            bz_loss, y_hat = model(user_titles, user_abstracts, log_mask, news_titles, news_abstracts)
            loss += bz_loss.data.float()
            accuracy += utils.acc(targets, y_hat)
            cnt += 1

        # Metric calculations
        auc = roc_auc_score(targets.cpu().numpy(), y_hat.cpu().numpy())
        mrr = mrr_score(targets.cpu().numpy(), y_hat.cpu().numpy())
        ndcg5 = ndcg_score(targets.cpu().numpy(), y_hat.cpu().numpy(), k=5)
        ndcg10 = ndcg_score(targets.cpu().numpy(), y_hat.cpu().numpy(), k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

    # Log final results
    logging.info(f"Mean AUC: {np.mean(AUC)}")
    logging.info(f"Mean MRR: {np.mean(MRR)}")
    logging.info(f"Mean nDCG@5: {np.mean(nDCG5)}")
    logging.info(f"Mean nDCG@10: {np.mean(nDCG10)}")

    logging.info(f'Test_loss: {loss.data / cnt}, Test_acc: {accuracy / cnt}')


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    utils.dump_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args.model_dir = '../model'
    args.model = 'NRMS_abstract'
    args.mode = 'test'
    args.load_ckpt_name = 'epoch-1.pt'
    args.enable_gpu = False
    args.use_category = False
    args.use_subcategory = False
    args.prepare = True

    if args.word_embedding_type == 'bert':
        args.word_embedding_dim == 768

    if 'train' in args.mode:
        if args.prepare:
            logging.info('Preparing training data...')
            total_sample_num = prepare_training_data(args.train_data_dir, args.nGPU, args.npratio, args.seed)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.train_data_dir, f'behaviors_np{args.npratio}_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(
                        f'Splited training data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()
                result = subprocess.getoutput(f'wc -l {data_file_path}')
                total_sample_num += int(result.split(' ')[0])
            logging.info('Skip training data preparation.')
        logging.info(
            f'{total_sample_num} training samples, {total_sample_num // args.batch_size // args.nGPU} batches in total.')

        if args.nGPU == 1:
            train(None, args)
        else:
            torch.multiprocessing.spawn(train, nprocs=args.nGPU, args=(args,))

    if 'test' in args.mode:
        if args.prepare:
            logging.info('Preparing testing data...')
            total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)
        else:
            total_sample_num = 0
            for i in range(args.nGPU):
                data_file_path = os.path.join(args.test_data_dir, f'behaviors_{i}.tsv')
                if not os.path.exists(data_file_path):
                    logging.error(
                        f'Splited testing data {data_file_path} for GPU {i} does not exist. Please set the parameter --prepare as True and rerun the code.')
                    exit()
                result = subprocess.getoutput(f'wc -l {data_file_path}')
                total_sample_num += int(result.split(' ')[0])
            logging.info('Skip testing data preparation.')
        logging.info(f'{total_sample_num} testing samples in total.')

        if args.nGPU == 1:
            test(None, args)
        else:
            torch.multiprocessing.spawn(test, nprocs=args.nGPU, args=(args,))

