from flask import *
from Controller import Recommendation,KmeansEvaluationController,CollaborativeEvaluationController,TransactionHistoryController

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home_page():

    data_set = {'Id' : '2', 'Fullname' : 'Hello Dennis Sebastian from default route!'}
    json_dump = json.dumps(data_set)

    return json_dump


@app.route('/Recommendation/',methods=['GET'])
def RecommendationCalc():
    filename = str(request.args.get('filename'))
    customer = str(request.args.get('customerShipTo'))
    result = Recommendation.CalculateRecommendation(filename, customer)
    return result

@app.route('/KMeansEvaluation/',methods=['GET'])
def KmeansEvaluationCalc():

    return result

@app.route('/CollaborativeEvaluation/',methods=['GET'])
def CollaborativeEvaluationCalc():

    return result


@app.route('/KMeansCFEvaluation/',methods=['GET'])
def KMeansCFEvaluationCalc():

    return result


@app.route('/TransactionHistory/',methods=['GET'])
def TransactionHistory():
    filename = str(request.args.get('filename'))
    customer = str(request.args.get('customerShipTo'))
    result = TransactionHistoryController.TransactionHistory(filename, customer)
    return result



app.run(port=8888,debug=True)