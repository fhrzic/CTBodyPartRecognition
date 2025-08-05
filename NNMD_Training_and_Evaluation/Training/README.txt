Scripts used to train models on the provided dataset. Usually it is focused to be trained on NNMD, but
there are also variations for other input data types.

* train_NNMD.py - naive traning on train:valid:test split.

* train_NNMD_on_fold.py - train model on train:test folds (number of fold is defined in script). The models 
			- are being trained one after another.

* train_NNMD_on_folds_multiporecessing.py - train model on train:test folds (number of fold is defined in script). The models 
			- are being trained in parallel. Be careful about available resources. Can throttle easily, 
			- recommended to use the script train_NNMD_on_fold.py

