Scripts which predicts data by totalSegmentator and export nifties and statistic

* run_TotalSegmentator_by_nifti.py - main script that does all in one run. The other two scripts separated this 
				     script into halfs because of possible errors and way to resume export
				   - INPUT IS nii.gz

* run_TotalSegmentator_by_nifti_no_xlsx.py - this applies TotalSegmentator to the files in provided dir, but it
					   - also check the exported files already. It skips them *cool feature)
					     . This is given as a parameter "input_dir_path" in the code

* run_generate_xlsx_from_ni.py - this script generates output xlsx file for all exported files in directory
			       - Store it in Output-labels.xlsx

* export_to_axialCT.py - scripts which simply convert any nii.gz to axial nii.gz.
					- that can also be done later, but this way it can be convinient