# ML-project-stability

Preliminaries
To run the code, you need to set two environment variables:

Set the DATASETS environment variable to a directory where datasets will be stored. For example: export DATASET="/my/directory/datasets".
Set the RESULTS environment variable to a directory where results will be stored. For example: export RESULTS="/my/directory/results".

The script gd.py trains a neural network using gradient descent. The required arguments are:
```
gd.py [dataset] [arch_id] [loss] [lr] [max_steps]
```

For example:
```
python src/gd.py cifar10-5k fc-tanh  mse  0.01 100000 --acc_goal 0.99 --neigs 2  --eig_freq 100
```

The above command will train a fully-connected tanh network (fc-tanh) on a 5k subset of CIFAR-10 (cifar10-5k) using the square loss (mse). We will run vanila gradient descent with step size 0.01 (lr). Training will terminate when either the train accuracy reaches 99% (train_acc) or when 100,000 (max_steps) iterations have passed. Every 50 (eig_freq) iterations, the top 2 (neigs) eigenvalues of the training loss Hessian will be computed and recorded. The training results will be saved in the following output directory:

```
${RESULTS}/cifar10-5k/fc-tanh/seed_0/mse/gd/lr_0.01
```

Within this output directory, the following files will be created, each containing a PyTorch tensor:

train_loss_final, test_loss_final, train_acc_final, test_acc_final: the train and test losses and accuracies, recorded at each iteration
eigs_final: the top 2 eigenvalues, measured every 50 (eig_freq) iterations