#WHEN PROF WILL give us the data the required probailities are computed using:

#1. Reload saved trained model: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
#2. Preprocess using MinMaxScaler
#3. Extract features using anova_column_name list
# #List of ANOVA-F selected feature names
#Index(['G3', 'G3_SMILES_in_Target Sequence_perc',
#       'G3_fdp_Target Sequence_base', 'G3_std_Target Sequence_dist', 'G9'],
#      dtype='object')
 
#4. Generating our predicted values with probabilities using trained model
# rnn_preda = model_trained.predict(x_test_data).ravel()
# print(rnn_preda)

from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. load the model
model = keras.models.load_model('gru_rnn_model_last.h5')

# 2. read blind test data provided
blind_data = pd.read_csv ('blind_test_data.csv', index_col='Index',)#header=None)
print("BLIND TEST DATA:")
print(blind_data.shape)
print(blind_data.head())

# 3. Retrieve the 5 ANOVA features only
blind_data_anova = blind_data[['G3', 'G3_SMILES_in_Target Sequence_perc','G3_fdp_Target Sequence_base','G3_std_Target Sequence_dist','G9']]
print("BLIND TEST DATA ANOVA FEATURES:")
print(blind_data_anova.shape)
print(blind_data_anova.head())

# 4. Preprocess using MinMaxScaler
scaler = MinMaxScaler()
print("YOOOOOOOOOO 2")
print(blind_data_anova.shape)
X_feat_t=scaler.fit_transform(blind_data_anova)
X_feat_s=pd.DataFrame(X_feat_t) #Scaled features
print("SCALED FEATURES:")
print(X_feat_s) 

# 5. Generating our predicted values with probabilities using trained model
x_test_data=np.array(X_feat_s)
print(x_test_data.shape)
x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],  x_test_data.shape[1], 1))
print(x_test_data.shape)
gru_rnn_preda = model.predict(x_test_data).ravel()
print(gru_rnn_preda)
print(type(gru_rnn_preda))

# 6. Unscale predictions
#scaler.inverse_transform(data_scaled)[:, [0]]
print("YOOOOOOOOOO 2")
print(gru_rnn_preda.shape)
gru_rnn_preda = np.reshape(gru_rnn_preda, (x_test_data.shape[0], 1))
unscaled_predictions = scaler.inverse_transform(gru_rnn_preda)


# 7. Write results to file
textfile = open("classification_results.txt", "w")
for element in unscaled_predictions.tolist():
    textfile.write(str(element) + "\n")
textfile.close()
