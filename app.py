import os
import random
import torch
import logging
import streamlit as st

from pathlib import Path
from src.prepare_data import prepare_training_data
from src.parameters import parse_args
from src.viz import ResultPredist

def main_function(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args.model_dir = "model"
    args.model = "NRMS"
    args.mode = "test"
    args.enable_gpu = False
    args.load_ckpt_name = "epoch-2.pt"
    if args.prepare:
        logging.info("Preparing training data...")
        total_sample_num = prepare_training_data(
            args.train_data_dir, args.nGPU, args.npratio, args.seed
        )

    results = ResultPredist(args)
    return results


st.title("News Recommendation Engine")

args = parse_args()
results = main_function(args)

user_id = st.number_input("Enter a user ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    result_dict = results.get_result(user_id)
    st.write("Candidate Documents:", result_dict["cand_doc"])
    st.write("Candidate Document Labels:", result_dict["cand_label"])
    st.write("User History:", result_dict["hostory"])
    st.write("AUC:", result_dict["auc"])
    st.write("MRR:", result_dict["mrr"])
    st.write("NDCG5:", result_dict["ndcg5"])
    st.write("NDCG10:", result_dict["ndcg10"])
    st.write("Scores:", result_dict["score"])

