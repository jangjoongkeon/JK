from flask import Flask, request, jsonify
import json
import os
from jsonschema import validate, ValidationError

from en.interface import SquadInference

SERVER_CONFIG_FILE = 'en/flask_config.json'
REQUEST_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'query': {'type': 'string', 'minLength': 1},
        'passages': {
            'type': 'array',
            'additionalItems': False,
            'minItems': 1,
            'uniqueItems': True,
            'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'id': {'type': 'string', 'minLength': 1},
                    'title': {'type': 'string', 'minLength': 1},
                    'content': {'type': 'string', 'minLength': 1}
                },
                'required': ['id', 'title', 'content']
            }
        }
    },
    'required': ['query', 'passages']
}
HTTP_BAD_REQUEST_CODE = 400

# Retrieve configuration
with open(SERVER_CONFIG_FILE) as f:
    config = json.load(f)

# Setup GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

# Create the interface
interface = SquadInference(
    config_file=config['bert_config_file'],
    vocab_file=config['bert_vocab_file'],
    lower_case=config['lower_case'],
    tmp_dir=config['tmp_dir'],
    model_ckpt=config['bert_checkpoint_file'],
    lr=3e-5,
    passage_max_len=config['context_max_len'],
    doc_stride=config['doc_stride'],
    question_max_len=config['question_max_len'],
    batch_size=config['batch_size'],
    n_best=config['n_best'],
    answer_max_len=config['answer_max_len'],
)

app = Flask(__name__)

@app.route('/mrc_en', methods=['GET'])
def event():
    # Retrieve the request
    request_data = request.get_json(force=True)

    # Check the format of the request
    try:
        validate(request_data, REQUEST_SCHEMA)
    except ValidationError as e:
        response = jsonify(e.message)
        response.status_code = HTTP_BAD_REQUEST_CODE
        return response

    # Predict
    response_data = interface.predict_request(request_data)
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host=config['host'], port=config['port'])
