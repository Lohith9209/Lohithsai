from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import random
import geocoder
import pycountry

app = Flask(__name__)

# IMPORT DATASETS
countries_pop = pd.read_csv(r'datasets/Countries_Population_final.csv')
countries_name = pd.read_csv(r'datasets/Countries_names.csv')

# Additional feature indicating population increase or decrease
def population_trend(country, last_year_pop, pred_pop):
    reasons_dict = {
        "Increase": ["Government Support", "High Number of Youth", 
                     "Medical Support", "Improved Healthcare Infrastructure",
                     "Higher Birth Rates", "Immigration", "Economic Growth", 
                     "Educational Opportunities", "Social Welfare Programs", "Technological Advancements"],
        "Decrease": ["Government Policies Restricting Births", "Aging Population", 
                     "Lack of Medical Support", "Natural Disasters", "Pandemics", 
                     "War and Conflict", "Economic Instability", "Emigration", 
                     "Environmental Degradation", "Resource Depletion", "Urbanization"]
    }
    if pred_pop > last_year_pop:
        trend = "Increase"
    elif pred_pop < last_year_pop:
        trend = "Decrease"
    else:
        trend = "No Change"
    
    reasons = reasons_dict.get(trend, [])
    if trend in ["Increase", "Decrease"]:
        # Randomly select 4 reasons
        selected_reasons = random.sample(reasons, min(4, len(reasons)))
        return trend, selected_reasons
    else:
        return trend, reasons

def get_current_location_country():
    try:
        # Using the 'geocoder' library to get location based on IP address
        g = geocoder.ip('me')
        
        if g.ok:
            country_code = g.country  # ISO 3166-1 alpha-2 country code
            country_name = pycountry.countries.get(alpha_2=country_code).name
            return country_name
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        country = request.form['country']
        year = request.form['year']
        
        if year.strip():  # Check if year is not empty
            if year.isnumeric():        
                # Divide Independent and Dependent features
                X = countries_pop['Year']
                y = countries_pop[country]

                # Train Test splitting
                X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)
                X_train = X_train.values.reshape(-1, 1)
                X_test = X_test.values.reshape(-1, 1)

                # POLYNOMIAL REGRESSION
                poly_features = PolynomialFeatures(degree=2)          
                X_train_poly = poly_features.fit_transform(X_train)          
                poly_model = LinearRegression() 
                poly_model.fit(X_train_poly, Y_train)     
                y_train_predicted = poly_model.predict(X_train_poly) 
                y_test_predict = poly_model.predict(poly_features.fit_transform(X_test)) 
                output = poly_model.predict(poly_features.fit_transform([[int(year)]]))           
        
                r2_test = r2_score(Y_test, y_test_predict)
    
                pred_pop = int(output[0])
                last_year_population = countries_pop[country].iloc[-1]
                trend, reasons = population_trend(country, last_year_population, pred_pop)
            
                # Prepare data for plotting
                years = countries_pop['Year']
                populations = countries_pop[country]
            
                return render_template('result.html', country=country.upper(), year=year, predicted_pop=pred_pop, trend=trend, reasons=reasons, accuracy=int(r2_test*100), years=years.tolist(), populations=populations.tolist())
            else:
                return "PLEASE ENTER A VALID YEAR"
        else:
            return render_template('error.html', message="Please enter a year.")
    else:
        return render_template('index.html', countries=sorted(countries_name['Country_Name']))

@app.route('/get_location', methods=['POST'])
def get_location():
    current_location_country = get_current_location_country()
    if current_location_country:
        return jsonify({'message': f'Your current location is in: {current_location_country}'})
    else:
        return jsonify({'message': 'Unable to fetch your current location. Please try again later.'})

if __name__ == '__main__':
    app.run(debug=True)


