from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model/scam_model.pkl')  # adjust path if needed

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            features = [
                int(request.form['username_length']),
                int(request.form['username_has_number']),
                int(request.form['full_name_has_number']),
                int(request.form['full_name_length']),
                int(request.form['is_private']),
                int(request.form['is_joined_recently']),
                int(request.form['has_channel']),
                int(request.form['is_business_account']),
                int(request.form['has_guides']),
                int(request.form['has_external_url']),
                int(request.form['edge_followed_by']),
                int(request.form['edge_follow'])
            ]

            prediction = model.predict([features])[0]
            if prediction == 1:
                result = "🚨 Fake Account Detected"
            else:
                result = "✅ Real Account"
        except Exception as e:
            result = f"⚠️ Error: {str(e)}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
