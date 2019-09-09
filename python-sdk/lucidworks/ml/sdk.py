import tempfile
import zipfile
import os
import importlib
import numpy as np
import numbers

PREDICTOR_PY_FILE = 'predict.py'


class LocalBundleRunner:
    """
    A class that runs Fusion Python ML bundles locally.

    The typical developer workflow would be to
        (1) train a model
        (2) create a Fusion Python ML bundle zip file
        (3) test locally using this class
        (4) deploy to Fusion.

    This class is used for quicker verification of the bundle zip file before proceeding to step (4)
    """

    def __init__(self, bundle_zip_file: str):
        """
        Create an instance of LocalBundleRunner

        :param bundle_zip_file: str, path to Fusion Python ML bundle ZIP file
        """
        self.bundle_zip_file = bundle_zip_file
        self.working_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(bundle_zip_file, 'r') as zip:
            zip.extractall(self.working_dir)

        assert os.path.exists(os.path.join(self.working_dir, PREDICTOR_PY_FILE)), "Bundle must contain predict.py file"

        self.predictor = _load_predictor(os.path.basename(self.bundle_zip_file), self.working_dir)

    def predict(self, X: dict) -> dict:
        """
        Execute prediction, ensuring that the user-defined predict.py follows required
        specifications.

        :param x:
        :return:
        """
        y = self.predictor.predict(X)
        self._verify_y(y)
        return y

    def _verify_y(self, y: dict):
        """
        Verify that y satisfies these conditions:

        - x is a dict with field/values
        - Should perform input validation, and throw ValueError for invalid input
        - Then perform any necessary preprocessing and prediction
        - Must return a dict, with field/values.  Returned values must be of one of the following types:
            - Any numbers.Number (float, int, etc.)
            - str
            - list or ndarray of str
            - list or ndarray of numbers.Number

        :param y:
        :return:
        """
        fields = []
        for key, val in y.items():
            if isinstance(val, np.ndarray):
                val = val.tolist()

            value = None
            if isinstance(val, numbers.Number):
                pass
            elif isinstance(val, str):
                pass
            elif isinstance(val, list):
                if val:
                    if isinstance(val[0], numbers.Number):
                        pass
                    elif isinstance(val[0], str):
                        pass
                    else:
                        raise RuntimeError("Unsupported type in list for field {}, type={}. "
                                           "Must be either a str or numbers.Number".format(key, type(val[0])))
            else:
                raise RuntimeError("Unsupported type for field {}, type={}. "
                                   "Must be one of (str, numbers.Number, list/ndarray of str, "
                                   "or list/ndarray of numbers.Number)".format(key, type(val)))


def _load_predictor(module_name: str, path: str):
    """
    Dynamically load predictor from path containing bundle

    :param path:
    :return:
    """
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(path, PREDICTOR_PY_FILE))
    predictor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predictor)
    predictor.init(path)

    return predictor


from requests.auth import HTTPBasicAuth
import requests
import json


class MLServiceSDKFusionClient:
    """
    REST client for Fusion's ML service, to aid in deploying and testing custom Python models
    for Fusion.

    This should only be used for testing and development purposes.  Not for production use.
    """

    _DEFAULT_HEADERS = {'Content-Type': 'application/json'}

    def __init__(self,
                 base_url: str,
                 fusion_app: str,
                 auth: HTTPBasicAuth = HTTPBasicAuth('admin', 'password123')):
        """
        Create an instance of MLServiceSDKFusionClient

        :param base_url:
        :param fusion_app:
        :param auth:
        """
        self.base_url = base_url
        self.fusion_app = fusion_app
        self.auth = auth

    def upload_model(self,
                     bundle_zip_file: str,
                     model_id: str,
                     model_type: str='python'):
        """
        Upload a model to Fusion ML service, replacing model if it already exists

        :param bundle_zip_file: Path to the Python ML bundle zip file
        :param model_id: Model ID which uniquely identifies model
        :return:
        """
        res = requests.post('{}/ai/ml-models?modelId={}&type={}'.format(self.base_url, model_id, model_type),
                            auth=self.auth,
                            files={'file': open(bundle_zip_file, 'rb')})
        res.raise_for_status()

    def delete_model(self, model_id):
        """
        Delete a model

        :param model_id:
        :return:
        """
        res = requests.delete('{}/ai/ml-models/{}'.format(self.base_url, model_id), auth=self.auth)
        res.raise_for_status()

    def list_models(self):
        """
        List all models in Fusion ML service.

        :return:
        """
        res = requests.get('{}/ai/ml-models'.format(self.base_url),
                           auth=self.auth,
                           headers=self._DEFAULT_HEADERS)
        res.raise_for_status()
        return res.json().get('models', [])

    def predict(self, model_id: str, X: dict) -> dict:
        """
        Use Fusion ML service to run prediction.

        :param X: dictionary containing input to model
        :return: prediction output, dictionary
        """
        res = requests.post("{}/ai/ml-models/{}/prediction".format(self.base_url, model_id),
                            data=json.dumps(X),
                            auth=self.auth,
                            headers=self._DEFAULT_HEADERS)
        if res.status_code != requests.codes.ok:
            raise RuntimeError("Error, HTTP status={}, response={}".format(res.status_code, res.text))
        return res.json()

    def create_sample_ml_query_pipeline(self,
                                        pipeline_id: str,
                                        model_id: str,
                                        model_input_field: str,
                                        model_output_field: str):
        """
        Creates or updates a sample query pipeline containing a Machine Learning stage that:
        - Maps the 'q' request parameter to a model input field with name "model_input_field"
        - Executes the prediction
        - Maps the model output field with name 'model_output_field' to a request parameter with
          the same name.

        :param pipeline_id: ID of query pipeline to create or update
        :param model_id: Model ID of model to execute in ML service
        :param model_input_field: Name of model input field to map
        :param model_output_field: Name of model output field to map
        :return:
        """
        input_script = """
var modelInput = new java.util.HashMap()
modelInput.put("{}", request.getFirstParam("q"))
modelInput""".format(model_input_field)

        output_script = """request.putSingleParam("{}", modelOutput.get("{}"))""".format(model_output_field,
                                                                                         model_output_field)
        self.create_ml_query_pipeline(pipeline_id, model_id, input_script, output_script)

    def create_ml_query_pipeline(self,
                                 pipeline_id: str,
                                 model_id: str,
                                 input_js_script: str,
                                 output_js_script: str):
        """
        Creates or updates a query pipeline containing a Machine Learning stage.

        :param pipeline_id: ID of query pipeline to create or update
        :param model_id: Model ID of model to execute in ML service
        :param input_js_script: Model input transformation JS script
        :param output_js_script: Model output transformation JS script
        :return:
        """

        pipeline = {
            'id': pipeline_id,
            'stages': [
                {
                    'failOnError': True,
                    'id': '1bj',
                    'licensed': True,
                    'modelId': model_id,
                    'inputScript': input_js_script,
                    'outputScript': output_js_script,
                    'skip': False,
                    'type': 'ml-query'
                },
                {
                    'licensed': True,
                    'queryFields': [],
                    'returnFields': ['*', 'score'],
                    'rows': 10,
                    'skip': False,
                    'start': 0,
                    'type': 'search-fields'
                },
                {
                    'allowFederatedSearch': False,
                    'httpMethod': 'POST',
                    'licensed': True,
                    'responseSignalsEnabled': True,
                    'skip': False,
                    'type': 'solr-query'
                }
            ]
        }

        res = requests.post('{}/apps/{}/query-pipelines'.format(self.base_url, self.fusion_app),
                            data=json.dumps(pipeline),
                            auth=self.auth,
                            headers=self._DEFAULT_HEADERS)
        if res.status_code == requests.codes.conflict:
            res = requests.put('{}/apps/{}/query-pipelines/{}'.format(self.base_url, self.fusion_app, pipeline_id),
                               data=json.dumps(pipeline),
                               auth=self.auth,
                               headers=self._DEFAULT_HEADERS)
            res.raise_for_status()

            res = requests.put(
                '{}/apps/{}/query-pipelines/{}/refresh'.format(self.base_url, self.fusion_app, pipeline_id),
                auth=self.auth,
                headers=self._DEFAULT_HEADERS)
            res.raise_for_status()
        else:
            res.raise_for_status()

    def query(self,
              pipeline_id: str,
              collection_id: str,
              query: str,
              addl_params:dict={}):
        """
        Execute a query using specified query pipeline and collection

        :param pipeline_id:
        :param collection_id:
        :param query:
        :param addl_params:
        :return:
        """

        params = {
            'q': query,
            'wt': 'json'
        }

        params.update(addl_params)

        res = requests.get(
            '{}/apps/{}/query-pipelines/{}/collections/{}/select'.format(self.base_url, self.fusion_app, pipeline_id,
                                                                         collection_id),
            params=params,
            auth=self.auth,
            headers=self._DEFAULT_HEADERS)

        if res.status_code != requests.codes.ok:
            raise RuntimeError("Error, HTTP status={}, response={}".format(res.status_code, res.text))

        return res.json()
