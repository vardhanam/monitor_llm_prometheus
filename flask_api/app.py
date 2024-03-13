from flask import Flask, request, Response, stream_with_context
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from flask_prometheus_metrics import register_metrics
from ollama import Client

def create_app():
    app = Flask(__name__)

    # Initialize the Ollama Client outside of the request to avoid reinitialization on every request
    client = Client(host='http://localhost:11434')

    @app.route('/generate', methods=['POST'])
    def generate():
        # Get the prompt from the request's data
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return Response("No prompt provided", status=400)

        # Streaming the response using Ollama
        def generate_stream(prompt):
            try:
                for token in client.chat(model='llama2:7b', messages=[{'role': 'user', 'content': prompt}], stream=True):
                    # Use yield to stream the response
                    yield token['message']['content']
            except Exception as e:
                # Stop the stream and handle the exception if necessary
                yield str(e)
                return

        # The stream_with_context decorator is necessary to keep the context around the generator alive
        return Response(stream_with_context(generate_stream(prompt)), content_type='text/plain')

    # provide app's version and deploy environment/config name to set a gauge metric
    register_metrics(app, app_version="v0.1.2", app_config="staging")

    # Plug metrics WSGI app to your main app with dispatcher
    dispatcher = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})

    return app

if __name__ == '__main__':
    # This is used when running locally only. When deploying to a WSGI server, the application
    # is served by the WSGI server and not this run_simple command.
    from werkzeug.serving import run_simple
    app = create_app()
    run_simple(hostname="0.0.0.0", port=5000, application=app.wsgi_app, threaded=True)