from flask import Flask, render_template
from controllers.upload_controller import upload_controller
from controllers.webcam_controller import webcam_controller

app = Flask(__name__)

# Register Controllers
app.register_blueprint(upload_controller)
app.register_blueprint(webcam_controller)


@app.route("/")
def index():
    return render_template("index.html", title="Flask")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
