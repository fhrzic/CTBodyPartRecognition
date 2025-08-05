There are the scripts for conversion of 3D - CT to 2D images.
* Supported formats: - Single folder containing DCMS
		     - Zip containing folder containing DCMS
                     - NIFTI to 2D 9 (nii.gz)

-----------------------------------------------------------------------------------------------------
* multiProcessExport.py - uses container to obtain nifty, and then exports exports 2D image out of it. It is like complete pipeline.
			- WORKS ON ZIP FILES (NNMD) 

-----------------------------------------------------------------------------------------------------

* exportOneFolder.py - is simplified version of multiProcessExport.py where it accepts one folder filled with dicoms and export founded series in nifty. Then it exports 2d images
		             - WORKS ON DIR WITH DICOM (BCH)

* export_BCH_CT_multiprocess - multiprocess script for exportOneFolder
				             - WORKS ON DIR WITH DICOM (BCH)

-----------------------------------------------------------------------------------------------------
* copy_images_to_file - copy "reducted.png" images from source dir to dst dir 

* copy_images_to_file_bch - copy "reducted.png" images from source dir to dst dir but it renames them so they are tractable for xlsx

* extractTargetedCases_BCH.py - script which also accepts filter list of body parts to be extracted. It works on BCH data 

* organize_cases_SAROS.py - TotalSegmentator export the prediction in the folder where it finds nii.gz file.
                            so in order to organize the folders in separate cases, this script can be used.
-----------------------------------------------------------------------------------------------------

* export_CTtoXray.py - input is dir which contains nii.gz files. It uses script to export 
	             	 - WORKS ON NIFTI FILES (SAROS)

* export_CTtoXRay_multiprocess.py - input is dir which contains nii.gz files. It uses script to export
								  - same as exoport_CTtoXray.py but supports multiprocess export
								  - WORKS ON NIFTI FILES (SAROS) 

-----------------------------------------------------------------------------------------------------
* ContarinerMinimumWorkingSample.ipynb - shows minimum example on how to read 3D image and generate nifty file from it. Also it extract necessary data of tags and allows tags selection.

* DevelopmentScript.ipynb - GARBAGE but might be usefull to debug stuff.