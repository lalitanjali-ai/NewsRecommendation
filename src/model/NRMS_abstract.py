import torch
from torch import nn
import torch.nn.functional as F

from .model_utils import AttentionPooling, MultiHeadSelfAttention


class NewsEncoder(nn.Module):
    def __init__(self, args, embedding_matrix):
        super(NewsEncoder, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.drop_rate = args.drop_rate
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn_title = MultiHeadSelfAttention(
            args.word_embedding_dim,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.multi_head_self_attn_abstract = MultiHeadSelfAttention(
            args.word_embedding_dim,
            args.num_attention_heads,
            self.dim_per_head,
            self.dim_per_head
        )
        self.attn_title = AttentionPooling(args.news_dim, args.news_query_vector_dim)
        self.attn_abstract = AttentionPooling(args.news_dim, args.news_query_vector_dim)
        self.combine_embeddings = nn.Linear(args.news_dim * 2, args.news_dim)

    def forward(self, x_title, x_abstract, mask_title=None, mask_abstract=None):
        '''
            x_title: batch_size, word_num_title
            x_abstract: batch_size, word_num_abstract
            mask_title: batch_size, word_num_title
            mask_abstract: batch_size, word_num_abstract
        '''
        word_vecs_title = F.dropout(self.embedding_matrix(x_title.long()),
                                    p=self.drop_rate,
                                    training=self.training)
        multihead_text_vecs_title = self.multi_head_self_attn_title(word_vecs_title, word_vecs_title, word_vecs_title,
                                                                    mask_title)
        multihead_text_vecs_title = F.dropout(multihead_text_vecs_title,
                                              p=self.drop_rate,
                                              training=self.training)
        news_vec_title = self.attn_title(multihead_text_vecs_title, mask_title)

        word_vecs_abstract = F.dropout(self.embedding_matrix(x_abstract.long()),
                                       p=self.drop_rate,
                                       training=self.training)
        multihead_text_vecs_abstract = self.multi_head_self_attn_abstract(word_vecs_abstract, word_vecs_abstract,
                                                                          word_vecs_abstract, mask_abstract)
        multihead_text_vecs_abstract = F.dropout(multihead_text_vecs_abstract,
                                                 p=self.drop_rate,
                                                 training=self.training)
        news_vec_abstract = self.attn_abstract(multihead_text_vecs_abstract, mask_abstract)

        combined_news_vec = torch.cat((news_vec_title, news_vec_abstract), dim=-1)
        news_vec = self.combine_embeddings(combined_news_vec)

        return news_vec


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.dim_per_head = args.news_dim // args.num_attention_heads
        assert args.news_dim == args.num_attention_heads * self.dim_per_head
        self.multi_head_self_attn = MultiHeadSelfAttention(args.news_dim, args.num_attention_heads, self.dim_per_head,
                                                           self.dim_per_head)
        self.attn = AttentionPooling(args.news_dim, args.user_query_vector_dim)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)

    def forward(self, news_vecs, log_mask=None):
        '''
            news_vecs: batch_size, history_num, news_dim
            log_mask: batch_size, history_num
        '''
        bz = news_vecs.shape[0]
        if self.args.user_log_mask:
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs, log_mask)
            user_vec = self.attn(news_vecs, log_mask)
        else:
            padding_doc = self.pad_doc.unsqueeze(dim=0).expand(bz, self.args.user_log_length, -1)
            news_vecs = news_vecs * log_mask.unsqueeze(dim=-1) + padding_doc * (1 - log_mask.unsqueeze(dim=-1))
            news_vecs = self.multi_head_self_attn(news_vecs, news_vecs, news_vecs)
            user_vec = self.attn(news_vecs)
        return user_vec


class Model(torch.nn.Module):
    def __init__(self, args, embedding_matrix, num_categories, num_subcategories):
        super(Model, self).__init__()
        self.args = args
        if args.enable_gpu:
            embedding_matrix = torch.from_numpy(embedding_matrix).float().cuda()
        else:
            embedding_matrix = torch.from_numpy(embedding_matrix).float()
        word_embedding = nn.Embedding.from_pretrained(embedding_matrix,
                                                      freeze=args.freeze_embedding,
                                                      padding_idx=0)

        self.news_encoder = NewsEncoder(args, word_embedding)
        self.user_encoder = UserEncoder(args)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, history_title, history_abstract, history_mask, candidate_title, candidate_abstract, label):
        '''
            history_title: batch_size, history_length, num_word_title
            history_abstract: batch_size, history_length, num_word_abstract
            history_mask: batch_size, history_length
            candidate_title: batch_size, 1+K, num_word_title
            candidate_abstract: batch_size, 1+K, num_word_abstract
            label: batch_size, 1+K
        '''
        candidate_news_title = candidate_title.reshape(-1, self.args.num_words_title)
        candidate_news_abstract = candidate_abstract.reshape(-1, self.args.num_words_abstract)
        candidate_news_vecs = self.news_encoder(candidate_news_title, candidate_news_abstract).reshape(
            self.args.batch_size, -1, self.args.news_dim)

        history_news_title = history_title.reshape(-1, self.args.num_words_title)
        history_news_abstract = history_abstract.reshape(-1, self.args.num_words_abstract)
        history_news_vecs = self.news_encoder(history_news_title, history_news_abstract).reshape(self.args.batch_size,
                                                                                                 self.args.user_log_length,
                                                                                                 self.args.news_dim)

        user_vec = self.user_encoder(history_news_vecs, history_mask)
        score = torch.bmm(candidate_news_vecs, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        loss = self.loss_fn(score, label)
        return loss, score