Scripts that generates patches from given data. Required inputs: 
    - Each sample must have json, nii.gz, total_segmentor files (both), -reducted image
    - input dir - dir where all indices are stored
    - xlsx path - directory where the annotation result of the observed sample is stored
                - this can be obtained on several ways, one is by using model to predict it,
                  create dataloader which simply generates it for every image (NMDD dataset),


--------------------------------------------------------------------------------------------------
* DevelopmentScript.py - notebook for development, it can be ignored but it holds interesting stuff

--------------------------------------------------------------------------------------------------

* export_data.py - script for exporting data. It first extract biggest blobs for every region and then
                   merges them based on the given criteria. This can sometimes cause errors
                 - NOT MULTIPROCESSING

* MainGenerationScript.py - main script for exporting data. It first merges all the blobs based on 
                            selected criteria, and then it filters them and find the biggest ones.
                            This is the desired behaviour.
                          - MULTIPROCESSING SCRIPT

--------------------------------------------------------------------------------------------------

* generate_xlsx.py - script which generates xlsx in the given dir. It take case, take total segmentor xlsx
                     and applies desired reduction. (NMDD based dl)

* generate_json.py - this is not necessary for any other dataset except for SAROS. If the json file 
                     which cointains basic data (such as projection) is missing, then it must be added
                     with assumed axial projection.

--------------------------------------------------------------------------------------------------
