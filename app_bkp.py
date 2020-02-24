import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model.
with open(f'model/svm-classifier.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
app.debug = True
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':
		variance = flask.request.form['variance']
		skewness = flask.request.form['skewness']
		curtosis = flask.request.form['curtosis']
		entropy  = flask.request.form['entropy']

		input_variables = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=['variance', 'skewness', 'curtosis' ,'entropy'], dtype=float)

		#print(input_variables)
		prediction = model.predict(input_variables)[0]

		print(f"Prediction-{prediction}")
		return flask.render_template('main.html', result='True', original_input={
			'variance' : variance,
			'skewness' : skewness,
			'curtosis' : curtosis,
			'entropy'  : entropy
			},
			Prediction= prediction)
if __name__ == '__main__':
	app.run(debug=True)