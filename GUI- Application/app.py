from flask import Flask, render_template, url_for, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    feature_vector = []

    encode_col =['Side',
                'City',
                'County',
                'State',
                'Timezone',
                'Wind_Direction',
                'Weather_Condition',
                'Amenity',
                'Bump',
                'Crossing',
                'Give_Way',
                'Junction',
                'No_Exit',
                'Railway',
                'Roundabout',
                'Station',
                'Stop',
                'Traffic_Calming',
                'Traffic_Signal',
                'Turning_Loop',
                'Sunrise_Sunset',
                'Weekday']

    num_cols =['TMC',
                'Start_Lng',
                'Start_Lat',
                'Temperature(F)',
                'Humidity(%)',
                'Pressure(in)',
                'Visibility(mi)',
                'Hour']

    total_cols =['TMC', 'Start_Lng', 'Start_Lat', 'Side', 'City', 'County',
                'State', 'Timezone', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                'Visibility(mi)', 'Wind_Direction', 'Weather_Condition', 'Amenity',
                'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
                'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
                'Turning_Loop', 'Sunrise_Sunset', 'Hour', 'Weekday']

    for col in total_cols:
        if col in encode_col:
            with open(f"/home/rajkamal/Documents/major/encoders/{col}_encoder.pkl", "rb") as f:
                le = pickle.load(f)

            try:
                val =request.form[f"{col}"]
                val_ =""
                for word in val:
                    val_ += word
                dummy_list = []
                dummy_list.append(val_)
                gen_label =le.transform(np.array(dummy_list))
                feature_vector.append(gen_label)
            except Exception as e:
                print("This is an exception: ", e)
                feature_vector.append(0)

        elif col in num_cols:

            feature_vector.append(float(request.form[f"{col}"]))

    feature_vector = np.array(feature_vector)

    feature_vector =feature_vector.reshape((1, -1))

    with open("/home/rajkamal/Documents/major/Random_Forest.pkl", "rb") as f:
        RF_clf =pickle.load(f)

    prediction =RF_clf.predict_proba(feature_vector)[0]

    severity = np.argmax(prediction) + 1
    probability =prediction[np.argmax(prediction)] * 100

    print("This is the Final Prediction: ", prediction)
    print("This is the Severity Level predicted:", severity)
    print("This is the probability of having that severity level", probability)

    if severity == 1 and probability <= 50:
        comments ="Since the Severity Level is One and the Confidence Score is NOT above 50%, the Likelihood of occurrence of an Accident is LOW"
    elif severity == 1 and probability > 50:
        comments ="Since the Severity Level is One and the Confidence Score is above 50%, the Likelihood of occurrence of an Accident is MODERATE"

    elif severity == 2 and probability <= 50:
        comments ="Since the Severity Level is Two and the Confidence Score is NOT above 50%, the Likelihood of occurrence of an Accident is MODERATE"

    elif severity == 2 and probability > 50:
        comments ="Since the Severity Level is Two and the Confidence Score is above 50%, the Likelihood of occurrence of an Accident is MODERATE"

    elif severity == 3 and probability <= 50:
        comments ="Since the Severity Level is Three and the Confidence Score is NOT above 50%, the Likelihood of occurrence of an Accident is MODERATE"

    else:
        comments ="The Likelihood of occurrence of an Accident is HIGH"

    return render_template('dummy.html', severity_level= severity, probability= probability, comments =comments)


if __name__ == "__main__":
    app.run(debug=True)