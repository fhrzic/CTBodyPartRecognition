# CTBodyPartRecognition
GitHub Repository for the paper: "A simple and effective approach for body part recognition on CT scans based on projection estimation" with the preprint available at
[https://arxiv.org/abs/2504.21810](https://arxiv.org/abs/2504.21810)



# Introduction
The repository serves the purpose of repeating all experiments conducted in the research mentioned above. The repository consists of many directories, which will be briefly explained:

- **3D2D** -- Scripts for converting 3D nifty files to 2D images.</li>
- **DataGenerationScripts_3D** -- Scripts for converting 2D images back to 3D nifty files.</li>
- **DataGenerationScripts_BCH_and_verification** -- Scripts for labeling BCH data. This is manual labeling!</li>
- **DataGenerationScripts_NMDD** --This is a placeholder directory because the NMDID dataset was labelled withthe  help of NNs.</li>
- **DataGenerationScripts_PATCHES** -- Script which generates patches from the given data based on the segmentation masks.</li>
- **DataVerificationScirpts_PATCHES_AND_FULL** -- Scripts used to verify exported patches. For every 3D volume, several patches are generated. Each patch must be verified.</li>
- **Evaluate_Model_Performance_2D_NMDD_BCH_SAROS** -- Scripts to evaluate models' performance.</li>
- **FinalPipeline** -- Final pipeline which can be used to do out-of-the-box predictions.</li>
- **NNMD_Training_and_Evaluation** -- These are scripts which helped in labelling the NMDID dataset. Part of the DataGenerationScripts_NMDD</li>
- **TotalSegmentor_export** -- In order to help labelling, the foundation model TotalSegmetator is utilised to predict the segmentation maps. Segmentation maps are later used in the labelling process. Check [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) for more info.</li>
- **Train_Models_25D_NMDD_BCH_SAROS** -- Scripts for training 2.5D models.</li>
- **Train_Models_2D_NMDD_BCH_SAROS** -- Scripts for training 2D models.</li>
- **Train_Models_3D_NMDD_BCH_SAROS** -- Scripts for training 3D models.li>
- **Train_Models_Patches_NMDD_BCH_SAROS** -- Scripts for training 3D models on patches.</li>
- **UsefullScripts** -- Other scripts that might be useful regarding data manipulation or statistics</li> 

All scripts are documented and have a small README file to help users on how to use the code. For any assistance, contact fhrzic@uniri.hr

# Install
There is a ".txt" file named <requirements.txt> which contains the virtual environment requirements. In order to install it, run the following commands:

First create environment:
```
python -m venv new_env
source new_env/bin/activate 
```

Second, install requirements:
```
pip install -r requirements.txt
```
