from flask import Flask, render_template
import markdown as md
from flask_restful import reqparse, Api, Resource
import os
import sys
import traceback


project_root = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PORT = 8080

app = Flask(__name__)
api = Api(app)

# Load docs
content = ""
with open('get.md', 'r') as f:
    content = f.read()

stringOfMarkdown = md.markdown(content,
                               extensions=['markdown.extensions.attr_list',
                                           'markdown.extensions.codehilite',
                                           'markdown.extensions.fenced_code'])

# argument parsing
reqparser = reqparse.RequestParser()
reqparser.add_argument('data')


class Predictor(Resource):
    _model = None

    def get_or_create_model(self):
        """Get or create the model."""
        if self._model is None:
            self._model = lambda x: x

        return self._model

    def post(self):

        try:
            # use parser and find the user's query
            args = reqparser.parse_args()

            query_data = args['data']

            if query_data is None:
                raise KeyError("Your JSON input has no attribute 'data'")

        except KeyError:
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()

        # Load model (ONNX only)
        model = self.get_or_create_model()

        # Feed input to the model
        result = model(query_data)

        return result


@app.route('/', methods=['GET'])
def show_docs():
    return render_template('index.html', stringOfMarkdown=stringOfMarkdown)


if __name__ == "__main__":

    try:
        # Setup the Api resource routing here
        # Route the URL to the resource
        api.add_resource(Predictor, '/')

        print('Running server on port %s...' % DEFAULT_PORT)

        app.run(debug=False, host="0.0.0.0", port=DEFAULT_PORT)

    except Exception:
        traceback.print_exc()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(exc_traceback)
