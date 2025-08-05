Scripts used to train the 2D model on NMDD, BCH and SAROS datasets. 
The data can be located anywhere, but the paths to the datasets must be 
specified

------------------------------------------------------------------------

* Utils - Utility scripts containing:
		- conf_NMDD_BCH_SAROS.py (config file for datasets and training)
                - dataloader_NMDD_BCH_SAROS.py (dataloader for all data)
                - models_NMDD_BCH_SAROS.py (specification of a big number of possible models - not all work)
		- train_model_NMDD_BCH-SAROS.py (train model script which puts everything together)

------------------------------------------------------------------------

* train_NMDD_BCH_SAROS_DENSE161.py - script which train DenseNet161 on the NMDD, BCH, and SAROS datasets

* statistic_NMDD_BCH_SAROS.py - scripts which plots data distribution for the dl between train, valid and test sets

* evaluate_dl_NMDD_BCH_SAROS.py - this is a crazy good script which provides insight in everything!!! F1 score, FN, FP
				  for every bodypart in given dataloader. TODO is to make same for individual image.
                                  In that case see Full prediction pipeline

* obtain_predictions_dl_NMDD_BCH_SAROS.py - this script exports all predictions to an xlsx from dl. It runs model over
					    the data, obtain predictions, round them using threshold 0.5 and exports
					    true label to the output xlsx.

------------------------------------------------------------------------
