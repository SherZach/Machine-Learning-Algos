from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def root(): 
    print("Sucessful request")
    return "<html><h1>Hello world!</h1><p>How are you today?</p></html>"

@app.route('/test', methods=['POST'])
def test():
    # run model here from input
    # convert input to torch
    print(request)
    x = request.get_json()
    print(x)
    return "Success for test"

if __name__ == '__main__':
    app.run(host='localhost', port='4444')