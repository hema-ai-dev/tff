GTSRB Stage 1A IID Split

Split type:
    Stratified IID

Clients:
    5

Classes:
    43

Random seed:
    20260720

Files:
    centralized_train.csv
        Complete official training set for the Centralized baseline.

    global_test.csv
        Complete official test set shared by Centralized, Local-only,
        and FedAvg.

    clients/client_0_train.csv ... client_4_train.csv
        Per-client training manifests for Local-only and FedAvg.

    split_summary_long.csv
        One row per client/class pair.

    split_summary_wide.csv
        One row per client, with one column per class.

    split_metadata.json
        Reproducibility metadata and split fingerprint.

Important:
    Image files are not copied.
    Each CSV Path value is relative to:
        /home/hema/projects/tff_mnist_fedavg/data/gtsrb
