# Data and plotting

Base network accuracies are stored in `bases_dataframe.pkl`.
Pruned and re-trained network accuracies are stored in`paths/pruned_dataframe.pkl`.
Useful constants are stored in `common.py`.
Of particular note are the re-training methods: `lottery` means rewinding the weights and learning rate, `lr_lottery` means rewinding just the weights, `lr_finetune` means rewinding just the learning rate, `finetune` is standard fine-tuning, and `reinit` is reinitializing and re-training.

For instance, the following Python code will print the mean and standard deviation of the test accuracy of all the networks we trained:
```
import pickle
import common

with open('bases_dataframe.pkl', 'rb') as f:
    base_df = pickle.load(f)

print(base_df.groupby('network').agg({'test_acc': ['mean', 'std']}))
```

And the following Python code will print the average test accuracy achievable at each sparsity by iteratively rewinding the learning rate on a ResNet-50 for 90 epochs:
```
import pickle
import common

with open('pruned_dataframe.pkl', 'rb') as f:
    df = pickle.load(f)

print(df[
    (df['network'] == common.RESNET50)
    & (df['is_iterative'])
    & (df['retrain_method'] == common.FINETUNE_HIGH_LR)
    & (df['retrain_time'] == 90)
].groupby('density').agg({'test_acc': 'median'}))
```

To generate the figures in the paper, run `./gen.py`, which reads from the pandas dataframes in the directory.
