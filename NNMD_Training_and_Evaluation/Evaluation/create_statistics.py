from Utils.evaluate_xlsx import *

_my_container = statisticContainer("/home/franko/Desktop/BodyPartTraining/Evaluation/cheatsheet_names.xlsx")

# Folds
if True:
  for _fold_num in [0,1,2,3,4]:
        _my_container.load_dir(f"/home/franko/Desktop/BodyPartTraining/Evaluation/Output_Folds/Fold_{_fold_num}")
        _my_container.generate_statistics()
        _my_container.dump_data(f"fold_{_fold_num}")
        _my_container.export_to_xlsx("4kFolds_statistic.xlsx")
else:
    _my_container.load_dir(f"/home/franko/Desktop/BodyPartTraining/Evaluation/Output_4k")
    _my_container.generate_statistics()
    _my_container.dump_data(f"4kTraining")
    _my_container.export_to_xlsx("4kTraining_statistic.xlsx")


