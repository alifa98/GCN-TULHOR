# main.py

import gc
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from model_components import BertSimpleModel
from data_utils import BertSimplePreprocessor, BertSimpleTULPreprocessor
from training import BertTrainer, BertTrainerClassification
from graph_utils import build_spatial_graph, normalize_adjacency, scipy_to_torch_sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
DATA_PATH = "ho_geolife_res6.csv"
BATCH_SIZE = 1024
EMB_SIZE = 256
HIDDEN_SIZE = 256
EPOCHS_MLM = 5
EPOCHS_CLASSIFY = 150
NUM_HEADS = 4

def main():
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    # Load and preprocess data
    print("📥 Loading data...")
    full_df = pd.read_csv(DATA_PATH).rename(columns={"taxi_id": "user_id"})
    full_df["higher_order_trajectory_len"] = full_df["higher_order_trajectory"].str.split().apply(len)

    # Keep users with more than 2 records
    valid_users = full_df["user_id"].value_counts()[lambda x: x > 2].index
    full_df = full_df[full_df["user_id"].isin(valid_users)].reset_index(drop=True)

    # Trim trajectories to 90th percentile
    max_len = int(full_df["higher_order_trajectory_len"].quantile(0.9))

    def truncate(seq): return " ".join(seq.split()[:max_len])
    full_df["higher_order_trajectory"] = full_df["higher_order_trajectory"].apply(truncate)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        full_df.drop(columns=["user_id"]),
        full_df["user_id"],
        test_size=0.33,
        stratify=full_df["user_id"],
        random_state=42
    )
    X_train["user_id"] = y_train
    X_test["user_id"] = y_test

    # MLM Dataset (BERT-style masked pretraining)
    print("📚 Preparing MLM dataset...")
    ds_mlm = BertSimplePreprocessor(full_df)

    # Classification Datasets
    print("📚 Preparing classification datasets...")
    ds_classify_full = BertSimpleTULPreprocessor(full_df, hex_vocab=ds_mlm.vocab_hex)
    ds_classify_train = BertSimpleTULPreprocessor(X_train, hex_vocab=ds_mlm.vocab_hex, user_vocab=ds_classify_full.user_vocab)
    ds_classify_test = BertSimpleTULPreprocessor(X_test, hex_vocab=ds_mlm.vocab_hex, user_vocab=ds_classify_full.user_vocab)

    # Build Graph (H3 + trajectory edges)
    print("🌐 Building spatial graph...")
    ds_mlm.df["Original_seq_length"] = full_df["higher_order_trajectory"].str.split().apply(len)
    trajectory_seqs = ds_mlm.df[ds_mlm.df["Original_seq_length"] >= 2]["indices"].tolist()

    graph = build_spatial_graph(
        vocab_size=len(ds_mlm.vocab_hex),
        h3_edges=ds_mlm.pairEdgesMapped,
        trajectory_sequences=trajectory_seqs
    )
    adj_matrix = normalize_adjacency(nx.adjacency_matrix(graph))
    adj_torch = scipy_to_torch_sparse(adj_matrix).to(device)

    # Prepare vocabulary tensor
    X_vocab = torch.tensor([i for i in range(len(ds_mlm.vocab_hex))]).to(device)

    # Initialize model
    print("🧠 Initializing BERT-GCN model...")
    model = BertSimpleModel(
        vocab_size=len(ds_mlm.vocab_hex),
        input_dim=EMB_SIZE,
        output_dim=HIDDEN_SIZE,
        attention_heads=NUM_HEADS,
        user_size=len(ds_classify_full.user_vocab),
        mode=0
    ).to(device)

    # MLM Training
    print("🔁 Training MLM (pretraining)...")
    mlm_trainer = BertTrainer(
        model=model,
        dataset=ds_mlm,
        log_dir=None,
        checkpoint_path=None,
        batch_size=BATCH_SIZE,
        lr=1e-5,
        epochs=EPOCHS_MLM,
        print_every=BATCH_SIZE,
        accuracy_every=BATCH_SIZE
    )
    mlm_trainer(X_vocab, adj_torch)

    # Switch to classification mode
    print("🎯 Switching to classification mode...")
    model.change_mode(1)

    # Classification Training
    print("🔁 Training user classifier...")
    classify_trainer = BertTrainerClassification(
        model=model,
        train_ds=ds_classify_train,
        test_ds=ds_classify_test,
        batch_size=BATCH_SIZE,
        lr=2e-5,
        epochs=EPOCHS_CLASSIFY,
        print_every=BATCH_SIZE // 2
    )
    classify_trainer(X_vocab, adj_torch)

if __name__ == "__main__":
    main()
