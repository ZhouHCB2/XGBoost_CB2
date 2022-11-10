CB2 protein has two conformations, among which we choose 6KPF as the agonistic conformation and 5ZTY as the antagonistic conformation
We used Schrodinger for docking (some compounds were not successfully docked in batch docking and were deleted)

1.External validation set: BdbClass1698.csv

2.smiles.sdf convert smiles in external validation set to sdf file

3.ligprep was calculated by the ligand preparation module in Schrodinger, and the parameters are default

4.the proteins were preprocessed by protein preparation wizard moudle

5.**.zip files records the information of the docking box which calculated by receptor grid generation in Schrodinger

6.glide-dock_** files records the docking information

7.**_docked.csv records the docking results

8.Untitled.ipynb a records the code for drawing the docking result into a ROC curve

9.docked_ROC.tiff: ROC plot of docking prediction results 