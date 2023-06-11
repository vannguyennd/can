# CAN
These are the source code samples for reproducing the experimental results mentioned in our paper "
Code Action Network for Binary Function Scope Identification". Please refer to https://link.springer.com/chapter/10.1007/978-3-030-47426-3_55 for details.

## Data sets
We use the data sets mentioned in the paper (Shin et al., 2015) consisting of 2,200 different binaries including 2,064 binaries obtained from the findutils, binutils, and coreutils packages and compiled with both icc and gcc for Linux at four optimization levels O0, O1, O2, and O3. The remaining binaries forWindows are from various well-known open-source projects which were compiled with Microsoft Visual Studio for the x86 (32-bit) and the x86-64 (64-bit) architectures at four optimization levels Od, O1, O2, and Ox.

## Folder structure
At each level in terms of byte and machine instruction level, we create folders corresponding to the architectures for both Linux and Windows systems at x86 (32-bit) and x64 (64-bit).  For example, the folder named can_mi_level_pex86 (it means using CAN at machine instruction level at Windows 32 bit).

## Training, validating and testing process
In each folder (model), to train the model, we run the file *_train.py and then get the results for function start and function end in terms of Recall, Precision and F1 score for both validation and testing set.  We will base the best result in terms of F1 score on the task of the function end identification on the validation set to obtain the best-trained model and the best results for the testing set. Noting that the best results are often in some last steps (epochs). For example, to the model can_mi_level_pex86, after the training process, we have the best result in terms of F1 score on the task of the function end identification on the validation set at the time step 19500. We based on this one to obtain the best-saved model and the best results for the testing set.

Run *_bound_scope.py to get the results for the function boundary and function scope identification in terms of Recall, Precision and F1 score corresponding to the best results obtained from the training, validating and testing process. For example, from the training process, we know the best results for the testing set at the time steps 19500, we run the file *_bound_scope.py with this time steps value (19500) to get the corresponding results for the function boundary and function scope identification.

## Implementation
We implemented the Code Action Networks in Python using TensorFlow (version 1.6), an open-source software library for Machine Intelligence developed by the Google Brain Team. We ran our experiments on an Intel Xeon Processor E5-1660 which has 8 cores at 3.0 GHz and 128 GB of RAM.

## The model configuration
Please read the supplementary material of our paper for the detail of the model configuration.

## Citation

If you reference our paper (and/or) use our source code samples in your work, please kindly cite our paper.

@article{vannguyen-can-2020,<br/>
  author = {Van Nguyen, Trung Le, Tue Le, Khanh Nguyen, Olivier de Vel, Paul Montague, John Grundy and Dinh Phung},<br/>
  title = {Code Action Network for Binary Function Scope Identification},<br/>
  publisher = {Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},<br/>
  year = {2020},<br/>
  url = {https://link.springer.com/chapter/10.1007/978-3-030-47426-3_55}<br/>
}
