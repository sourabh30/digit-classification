from flask import Flask, request

from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {escape(username)}'


@app.route('/sum/<x>/<y>')
def sum_two_num(x,y):
    return f'Sum of {x} and {y} = {int(x) + int(y)}'

@app.route('/sum_int/<int:x>/<int:y>')
def sum_two_num_int(x,y):
    return f'Sum of two interger {x} and {y} = {x + y}'

@app.route('/sum_numbers', methods = ['POST'])
def sum_num():
    data = request.get_json( )
    x = data['x']
    y = data['y']
    sum = int(x) + int(y)
    print(sum)
    return str(sum)