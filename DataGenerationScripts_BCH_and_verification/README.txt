This scripts are used to do annotations with help of different scripts.

--------------------------------------------------------------------------------------------------

* MainScript.ipynb - notebook for data mainpulation, merging based on different labels and statistics
                   - there is a statistics script which allows you to check statistics of the data and evaluate
                     the "model output" without running the models. It requiers to have a xlsx file for each sample with
                     columns "predicted" and "<key_given_name>" 
                   - DATA GENERATION CAN BE NEGLECTED

* WidgetAnnonationScript.ipyn - notebook to verify labels for BCH dataset. It requiers exported labels either in CSV or xlsx
                              - Widget loads preselected labels and show image for annotator to annotate
                              - Has several options to save, to see previous and next image... intuitive
                              - OVERRIDES output dir

--------------------------------------------------------------------------------------------------

