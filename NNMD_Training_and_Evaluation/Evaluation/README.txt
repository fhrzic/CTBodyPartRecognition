Scripts created for models' evaluation. Main difference is in how the optimization during evaluation process is 
being done.

* evaluate.py - script which predicts on one image at the time. Really good for small number of images.

* evaluate_folds.py - script which predicted one image at the time but for all folds. Relatively slow

* evaluate_NNMD.py - script which evaluates the model. JUST STATISTICS. It does it on train, valid, test principle

* evaluate_ours.py - script which predicts on "our" dataset (similar to evaluate.py but different parsing of input data)

* evaluate_single.py - script for predicting on the single image. Nothing special

* evaluate_fold_dl.py - rapid prediction on the folds because it uses them as dl and not single files.

* create_statistics.py - create statistics and export it to the folder. Like proper statistics f1 score, accuracy etc.