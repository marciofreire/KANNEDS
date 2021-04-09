# KANNEDS - Using Kinematics Analysis and Neural Networks for Early Detection of Sepsis

These are the LSTM Neural Network source code and a sample dataset of the KANNEDS project. 

The code uses the files below and shows a 5-fold cross validation of the LSTM neural network with the Kinematics Features (KF) as input and with Vital Signs (VS) as input:
-	KANNEDS_NEGATIVES_SAMPLE_DATA_FINAL.csv
   - Sample dataset containing 1431 patients negative for sepsis
-	KANNEDS_POSITIVES_SAMPLE_DATA_FINAL.csv
   -	Sample dataset containing 877 patients positive for sepsis
-	KANNEDS_VARIABLES_FINAL.csv
   -	A table containing the clinical variables used.
   
To run the code, please proceed as follow:
1.	Download the files and put them in the same folder;
2.	Install a Python v3.6.9 environment with Keras v2.2.4 and Tensorflow v1.12 packages;
3.	Run the Python code. 
