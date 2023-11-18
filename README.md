# MTR-Metro-Passenger-Prediction

The subject of this project is the prediction of the passenger number per hour in the metro. The input data is the date, weather, rainfall and other natural factors, as well as human factors such as holidays. The input data were previously digitally converted, data cleaned and standardised. Also, considering that the current number of passengers may be correlated with the number of passengers in the previous 12 hours, in this project, not only the normal MNN was used for machine learning, but also the RNN with the addition of the time dimension was used for testing and comparing which of the two types of networks performs better for this dataset.

In this repository:
1. projet.py contains the main script for MNN model.
2. projet_RNN.py contains the maisn script for RNN and LSTM model.
3. Reduced.csv is used as dataset. However, if a better performance and generalization are required, the two datasets above contain more data for training.
