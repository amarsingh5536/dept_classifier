from flask import request, jsonify, views
from datetime import datetime
from . import app
from .predictors import predict_dept


class PredictDepartmentView(views.MethodView):
    """
    APIs Methods
    -----------
    GET: Return predicted department name as per given description.
    """
    def get(self):
        try:
            """
            @param: description > "Salary not credited"
            """
            if not request.args.get('description'):
                return jsonify({"error": "Param 'description' is required!"}), 400

            data = predict_dept(request.args.get('description'))
            response = jsonify({"message": ("Success"), "data": data})
            response.status_code = 200

        except Exception as error:
            response = jsonify({"error": "Something Went Wrong!"}), 400
        return response