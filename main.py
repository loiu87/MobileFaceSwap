import os

import flask
from image_test import image_test

# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = flask.Flask(__name__)


@app.get("/")
def hello():
    """Return a friendly HTTP greeting."""
    source_img_path = flask.request.args.get("source")
    target_img_path = flask.request.args.get("target")
    # print(source_img_path, target_img_path)
    # image_test(source_img_path, target_img_path, output_dir='results', image_size=224, merge_result=True, need_align=True, use_gpu=True)
    return f"Hello !\n"


if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
    