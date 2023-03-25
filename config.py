from os.path import join
from utils import seed_random_generators

# Seed random generators (python, torch, numpy)
seed_random_generators()

# Paths
data_path = "data"
figs_path = "figs"
docs_path = "docs"
checkpoints_path = "checkpoints"

# Filenames
images_file = join(data_path, "X.npy")
labels_file = join(data_path, "y.npy")

val_fraction = 0.20                 # Fraction of data to use for validation data
test_fraction = 0.40                # Fraction of data to use for test data

# Training 
num_workers = 8                     # Number of workers to use with dataloader.
device = "cuda:4"                      # Device for PyTorch to use. Can be "cpu" or "cuda:n".

# Baseline CNN model
config_cnn = {
                "name" : "CNN", 
                "batch_size" : 16,
                "lr" : 1e-4,
                "epochs" : 10,
                "val_per_epoch" : 4,
                "checkpoint_file" : join(checkpoints_path, "cnn.pth"),
                "loss_plot_file" : join(figs_path, "cnn_loss_plot.png"),
                "earlystop_limit" : 10
            }
