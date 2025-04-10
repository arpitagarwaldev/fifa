from flask import Flask, request, render_template, flash
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
import sys



application = Flask(__name__)
application.secret_key = 'your-secret-key-here'

app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_datapoint', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get form data with validation
            height = request.form.get('height')
            weight = request.form.get('weight')
            age = request.form.get('age')
            
            if not all([height, weight, age]):
                flash('Please fill in all required fields', 'danger')
                return render_template('home.html')
            
            try:
                height = float(height)
                weight = float(weight)
                age = int(age)
            except ValueError:
                flash('Invalid input: Height and Weight should be numbers, Age should be a whole number', 'danger')
                return render_template('home.html')
                
            data = CustomData(
                height=height,
                weight=weight,
                age=age,
                ball_control=float(request.form.get('ball_control')),
                dribbling=float(request.form.get('dribbling')),
                slide_tackle=float(request.form.get('slide_tackle')),
                stand_tackle=float(request.form.get('stand_tackle')),
                aggression=float(request.form.get('aggression')),
                reactions=float(request.form.get('reactions')),
                att_position=float(request.form.get('att_position')),
                interceptions=float(request.form.get('interceptions')),
                vision=float(request.form.get('vision')),
                composure=float(request.form.get('composure')),
                crossing=float(request.form.get('crossing')),
                short_pass=float(request.form.get('short_pass')),
                long_pass=float(request.form.get('long_pass')),
                acceleration=float(request.form.get('acceleration')),
                stamina=float(request.form.get('stamina')),
                strength=float(request.form.get('strength')),
                balance=float(request.form.get('balance')),
                sprint_speed=float(request.form.get('sprint_speed')),
                agility=float(request.form.get('agility')),
                jumping=float(request.form.get('jumping')),
                heading=float(request.form.get('heading')),
                shot_power=float(request.form.get('shot_power')),
                finishing=float(request.form.get('finishing')),
                long_shots=float(request.form.get('long_shots')),
                curve=float(request.form.get('curve')),
                fk_acc=float(request.form.get('fk_acc')),
                penalties=float(request.form.get('penalties')),
                volleys=float(request.form.get('volleys')),
                gk_positioning=float(request.form.get('gk_positioning')),
                gk_diving=float(request.form.get('gk_diving')),
                gk_handling=float(request.form.get('gk_handling')),
                gk_kicking=float(request.form.get('gk_kicking')),
                gk_reflexes=float(request.form.get('gk_reflexes')),
                country=request.form.get('country'),
                club=request.form.get('club')
            )

            pred_df = data.get_data_as_dataframe()
            print(pred_df)

            print('Debug: Data frame before prediction:')
            print(pred_df)
            
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            predicted_value = results[0]
            
            print('Debug: Prediction result:', predicted_value)
            print('Debug: Type of prediction:', type(predicted_value))
            
            formatted_value = f'${predicted_value:,.2f}'
            flash(f'Prediction successful! Value: {formatted_value}', 'success')
            return render_template('home.html', results=formatted_value)
        
        except Exception as e:
            error = CustomException(e, sys)
            flash(f'Error during prediction: {str(error)}', 'danger')
            return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
