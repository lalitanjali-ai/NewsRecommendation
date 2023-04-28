import os
import random
import torch
import logging
import streamlit as st
import utils

from pathlib import Path
from prepare_data import prepare_testing_data
from parameters import parse_args
from viz import ResultPredist

def setup():
    utils.setuplogger()
    args = parse_args()
    utils.dump_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args.model_dir = "src/model"
    args.model = "NRMS"
    args.mode = "test"
    args.enable_gpu = False
    args.load_ckpt_name = "epoch-1.pt"
    return args

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def prepare_and_load(args):
    if args.prepare:
        logging.info("Preparing training data...")
        total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)

    results = ResultPredist(args)
    return results

if __name__ == "__main__":
    args = setup()
    results = prepare_and_load(args)

    st.title("News Recommendation Engine")

    user_id = st.number_input("Enter a user ID:", min_value=1, step=1)

    if st.button("Get Recommendations"):
        result_dict = results.get_result(user_id)

        st.header("User History and Candidate Documents")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("User History")
            for doc in result_dict["hostory"]:
                st.write(doc)

        with col2:
            st.subheader("Candidate Documents")
            for doc in result_dict["cand_doc"]:
                st.write(doc)

        st.header("Candidate documents that were clicked")
        for i, label in enumerate(result_dict["cand_label"]):
            if label == 1:
                st.write(result_dict["cand_doc"][i])
                st.write("Scores:", result_dict["score"][i])

        st.header("Model Performance")
        st.write("AUC:", result_dict["auc"])
        st.write("MRR:", result_dict["mrr"])
        st.write("NDCG5:", result_dict["ndcg5"])
        st.write("NDCG10:", result_dict["ndcg10"])
        st.write("Scores:", result_dict["score"])

# import os
# import random
# import torch
# import logging
# import streamlit as st
# import utils
#
# from pathlib import Path
# from prepare_data import  prepare_testing_data
# from parameters import parse_args
# from viz import ResultPredist
#
#
# if __name__ == "__main__":
#     utils.setuplogger()
#     args = parse_args()
#     utils.dump_args(args)
#     random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "8888"
#     Path(args.model_dir).mkdir(parents=True, exist_ok=True)
#     args.model_dir = "src/model"
#     args.model = "NRMS"
#     args.mode = "test"
#     args.enable_gpu = False
#     args.load_ckpt_name = "epoch-1.pt"
#     if args.prepare:
#         logging.info("Preparing training data...")
#         total_sample_num = prepare_testing_data(args.test_data_dir, args.nGPU)
#
#     results = ResultPredist(args)
#
#     st.title("News Recommendation Engine")
#
#     user_id = st.number_input("Enter a user ID:", min_value=1, step=1)
#
#     if st.button("Get Recommendations"):
#         result_dict = results.get_result(user_id)
#         st.write("Candidate Documents:", result_dict["cand_doc"])
#         st.write("Candidate Document Labels:", result_dict["cand_label"])
#         st.write("User History:", result_dict["hostory"])
#         st.write("AUC:", result_dict["auc"])
#         st.write("MRR:", result_dict["mrr"])
#         st.write("NDCG5:", result_dict["ndcg5"])
#         st.write("NDCG10:", result_dict["ndcg10"])
#         st.write("Scores:", result_dict["score"])
#
