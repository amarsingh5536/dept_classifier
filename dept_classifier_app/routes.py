from . import app
from .api import PredictDepartmentView

app.add_url_rule('/predict/department/', 
    view_func=PredictDepartmentView.as_view('predict_department'))
