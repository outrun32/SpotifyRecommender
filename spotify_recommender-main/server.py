from flask import Flask, render_template, request, flash, url_for, redirect

from flask_ngrok import run_with_ngrok

from ML_Predictor import set_results

app = Flask(__name__)
run_with_ngrok(app)

TEMPLATE_FILE = "index.html"



@app.route("/", methods=['GET', 'POST'])
def upload_username():
    if request.method == "POST" and request.form["username"] not in ["", " "]:
        set_results(request.form["username"])
    return render_template(TEMPLATE_FILE)


app.run()