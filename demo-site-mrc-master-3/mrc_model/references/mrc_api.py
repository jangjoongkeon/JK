import ujson as json
import tensorflow as tf
import numpy as np
import time
import pickle

from mrc.mrc_call import MRC
from mrc.mrc_call_util import get_logger
from service_locator import config
from flask import Flask, request, jsonify

app = Flask(__name__)

logger= get_logger(drop_file=True)

max_passage = int(config.get('mrc.max_passage'))
api_port = int(config.get('mrc.api_port'))
sess_mrc = MRC(max_passage)

@app.route('/mrc_answer', methods=['GET'])
def event():

    mrc_status = 0
    outputs = []

    try:
        passages = request.json.get('passages')
        query = request.json.get('query')

        if query == None or passages == None:
            raise Exception('None parameter')

        passage_id = [p['id'] for p in passages ]
        passage_list = np.array([p['content'] for p in passages])
        passage_num = len(passage_list)

    except Exception as e:
        mrc_status = 100
        error_message = str(e).replace('\n', '=-=')
        logger.error("[%s]-[%s]" % ('request_error', error_message))

    if mrc_status == 0 :

        result = sess_mrc.search(query, passage_list, passage_num)

        if result['result_code'] != 100 :
            mrc_status = 200
            error_message = result['errorMessage'].replace('\n', '=-=')
            logger.error("[%s]-[%s]-[%s]-[%s]" % ('mrc_error', query,
                                                    '-+-'.join(passage_list), error_message))


        if mrc_status == 0 :

            for idx in range(len(result['items'])) :

                start_idx = result['items'][idx]['start_idx']
                end_idx = result['items'][idx]['end_idx']
                confidence = float(result['items'][idx]['confidence'])  # numpy.float => float

                response = {'id' : passage_id[idx], 'start_offset' : start_idx,
                                'end_offset' : end_idx, 'score' : confidence}
                outputs.append(response)

    sorted_outputs = sorted(outputs, key=lambda response : response['score'], reverse=True)

    return jsonify(sorted_outputs)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=api_port)
