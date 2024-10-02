import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
import joblib
import plotly.graph_objects as go
from numerize import numerize

warnings.filterwarnings("ignore")

# IMPORT DATASETS
countries_pop = pd.read_csv(r'datasets/Countries_Population_final.csv')
countries_name = pd.read_csv(r'datasets/Countries_names.csv')

# List all available countries
available_countries = countries_pop.columns[1:]  # All columns except 'Year'
print("Available countries for prediction:")
for country in available_countries:
    print(country)

# USER INPUTS FOR PREDICTION
option = input("Enter the name of the country for which you want to predict the population: ")

if option not in available_countries:
    print("The selected country is not available. Please choose a valid country from the list.")
else:
    year = input("Enter the year for which you want to predict the population: ")

    if year.isdigit():
        year = int(year)

        # Divide Independent and Dependent features
        X = countries_pop['Year'].values.reshape(-1, 1)  # all the independent features are copied to X
        y = countries_pop[option].values  # the dependent feature is copied to y

        # Train Test splitting
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        # Polynomial Regression
        def create_polynomial_regression_model(degree, Yearin):
            poly_features = PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train)  # transforms the existing features to higher degree features.

            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, Y_train)  # fit the transformed features to Linear Regression    
            y_train_predicted = poly_model.predict(X_train_poly)  # predicting on training data-set
            y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))  # predicting on test data-set
            output = poly_model.predict(poly_features.fit_transform([[Yearin]]))

            r2_test = r2_score(Y_test, y_test_predict)

            print("#### ALGORITHM: POLYNOMIAL REGRESSION")
            print('#### ACCURACY: ' + str(int(r2_test * 100)) + '%')

            # Save the model
            joblib.dump(poly_model, 'polynomial_regression_model.pkl')
            joblib.dump(poly_features, 'polynomial_features.pkl')

            return output

        pred = create_polynomial_regression_model(2, year)

        # OUTPUT DETAILS
        pred_pop = numerize.numerize(pred[0])
        print("#### COUNTRY:  " + option.upper())
        print("#### YEAR:  " + str(year))
        print("#### PREDICTED POPULATION:  " + pred_pop)
    else:
        print('PLEASE ENTER A VALID YEAR')

    # Plotting the results
    if isinstance(year, int):
        # Load the model
        poly_model = joblib.load('polynomial_regression_model.pkl')
        poly_features = joblib.load('polynomial_features.pkl')

        # Predict using the loaded model
        new_pred = poly_model.predict(poly_features.fit_transform([[year]]))
        new_pred_pop = numerize.numerize(new_pred[0])

        print('#### ' + option.upper() + "'S  POPULATION")
        fig1 = go.Figure()
        # Create and style traces
        fig1.add_trace(go.Scatter(x=countries_pop['Year'], y=countries_pop[option], name="Previous Year's",
                                  line=dict(color='green', width=11)
                                  ))
        fig1.add_trace(go.Scatter(x=[year], y=[new_pred[0]],
                                  name='Predicted ' + str(year),
                                  mode='markers',
                                  marker_symbol='star',
                                  marker=dict(
                                      size=20,
                                      color='red'  # set color
                                  )
                                  ))
        fig1.show()
        print('The above plot shows the population of a country from 1960 to 2021, star represents the predicted population for the given year')

        input("Press Enter to close the plot...")
