# Laptop-Price-prediction-with-Reccomendations
Use of machine learning models, for predicting laptop prices based on specifications provided by the users. The Four models were used to measure accuracy: Decision Tree, Ridge Regression, Linear Regression, and Ensemble method (Random Forest + XGBRegressor).
 The Ensemble method generated an R2 score of 0.896 and MAE of 0.149 which is best among others.  
 
Prediction of price for new user inputs is done by applying user inputs on random forest model. 
In the proposed architecture, an approach is introduced that leverages web scraping to display information from Amazon.in ecommerce website after predicting laptop prices through machine learning. 

This methodology enables to gather data on laptop models, specifications and prices which helps to reccomend laptop for one who is seeking to buy laptop.

# How to run :
1. Uploaded .py and .pnyb both files, download as per your ide and requirement.
2. Keep laptop_data.csv file in the same directory or change the path while reading csv file in the code.
3. After training model, all the results of models will be displayed as M2 score and MAE(Mean Absolute Error).
4. At last step sample inputs are given for web scrapping on random forest model you can uncomment code for taking inputs and processing on real time data.
5. Web scrapping will display 5 laptops as reccomendations.

# Output :
![image](https://github.com/Kirankumar6251/Laptop-Price-prediction-with-Reccomendations/assets/47715507/04775731-067c-4776-9441-b23e585343de)
