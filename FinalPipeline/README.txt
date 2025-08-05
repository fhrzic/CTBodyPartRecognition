Scripts for prediction of the cases. Full pipeline means that the scripts that the scripts are
capable to accept "CT" -- dicom folders and export them to the 2D image and then also 
predict on them.

-----------------------------------------------------------------------------------------------

* predict_one_case.py - script which predicts one case (full pipeline)
                      - it accepts as argument path to the image or directory containing dcms in.
		                If the input is image, then it assumes that it is "reducted_image.png" and
                        predicts with the provided model the labels for given image. If the input is
                        dir, then the scripts finds all series, export them to folder, export 2D image,
                        and then makes prediction.
                      - it requires path to the model which is obtained by the train_NMDD_BCH_SAROS_Dense161.py
                        script.
                      - optionally it can be provided by refiend thresholds in json file. This json can be obtained
                        by scripts: obtain_preidctions_dl_NMDD_BCH_SAROS.py --> evaluate_outputs_NMDD_BCH_SAROS.py 
                        --> export_thresholds_to_json.py

-----------------------------------------------------------------------------------------------