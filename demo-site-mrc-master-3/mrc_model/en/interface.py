from en.run_squad_42maru_single import *
from en.run_squad_42maru_single import _get_best_indexes
from en.run_squad_42maru_single import _compute_softmax


class SquadInference:
    """
    User-friendly wrapper around the tensorflow model. This class represents an
    interface, loading a tensorflow model and allowing users to infer questions
    from passages using a a JSON format easier to understand than tensorflow.
    Please refer to following link for the JSON format specifications :
    http://192.168.0.39/projects/mrc-in-english/wiki/Api_with_sass

    Attributes:
        batch_size (int): Batch-size to use (for inference batches).
        bert_config (BertConfig): Loaded BERT configuration.
        tokenizer (FullTokenizer): Tokienizer used by BERT.
        model_fn (function): Model function of the model.
        estimator (Estimator): Estimator representing the model.
        tmp_dir (str): Location where to store temporary files if needed.
        passage_max_len (int): Maximum allowed size for a passage for the model.
        doc_stride (int): Size of overlapping text in case of too long passage.
            Some explanations can be found here : https://github.com/google-research/bert/issues/66.
        question_max_len (int): Maximum allowed size for a question for the 
            model.
        answer_max_len (int): Maximum allowed size for an answer for the model.
        case (bool): Case used by the BERT model (`True` for Uncased and 
            `False` for Cased version).
        n_best (int): Number of best predictions to keep at inference.
    """

    def __init__(self, config_file, vocab_file, lower_case, tmp_dir, model_ckpt, 
        passage_max_len, doc_stride, question_max_len, batch_size, n_best, 
        answer_max_len, lr=3e-5):
        """
        Constructor of SquadInference. This constructor will simply create the 
        model based on the arguments given. It is here that weights are loaded.

        Args:
            config_file (str): Location of the BERT configuration file to use.
            vocab_file (str): Location of the vocabulary file to use for BERT.
            lower_case (bool): Case to use for BERT (`True` = Uncased; `False` 
                = Cased).
            tmp_dir (str): Location of the folder where temporary file can be 
                stored.
            model_ckpt (str): Location of the checkpoint of the model to use
                for inference.
            passage_max_len (int): Maximum allowed size for a passage.
            doc_stride (int): Size of overlapping text in case of too long 
                passage. Some explanations can be found here : https://github.com/google-research/bert/issues/66.
            question_max_len (int): Maximum allowed size for a question.
            batch_size (int): Batch-size to use (for inference batches).
            n_best (int): Number of best predictions to keep at inference.
            answer_max_len (int): Maximum allowed size for an answer.
            lr (float, optional): Learning rate. This is not used at inference,
                but still needed as parameter, so it is optional argument. 
                Defaults to `3e-5`.

        Note:
            This constructor is implemented for using GPU (or CPU), but not TPU.
            It is however possible with minimal changes. Maybe something to do 
            in the future.        
        """
        self.batch_size = batch_size
        self.bert_config = modeling.BertConfig.from_json_file(config_file)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, 
            do_lower_case=lower_case)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=None,
            master=None,
            model_dir=tmp_dir,
            save_checkpoints_steps=1000,    # Not used (no training)
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=1000,
                num_shards=8,               # Not used (no TPU)
                per_host_input_for_training=is_per_host))

        self.model_fn = model_fn_builder(
            bert_config=self.bert_config,
            init_checkpoint=model_ckpt,
            learning_rate=lr,
            num_train_steps=None,
            num_warmup_steps=None,
            use_tpu=False,
            use_one_hot_embeddings=False)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=False,
            model_fn=self.model_fn,
            config=run_config,
            train_batch_size=32,    # Not used
            predict_batch_size=self.batch_size)
        
        self.tmp_dir = tmp_dir
        self.passage_max_len = passage_max_len
        self.doc_stride = doc_stride
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.case = lower_case
        self.n_best = n_best

    def predict_request(self, request_data):
        """
        Main function of this interface. Predict a given request and return a
        response. The request and the response are both treated as JSON (not
        actual JSON, but python-equivalent of JSON : `dict` and `list`). This
        function will treat the given data and predict the answer. Answer is 
        then treated in order to return a response with JSON format.

        Note : 
            For a description of the JSON format of the request and the 
            answer, please refer to this page : http://192.168.0.39/projects/mrc-in-english/wiki/Api_with_sass

        Args:
            request_data (dict): Contain all the data of the request, in a 
                JSON-like format. 

        Returns:
            dict: Response data, in a JSON-like format.

        See also:
            _predict()

        """
        requests = []
        for passage in request_data['passages']:
            requests.append(Request(
                id=passage['id'],
                title=passage['title'],
                passage=passage['content'],
                query=request_data['query'],
            ))
        
        responses = self._predict(requests)

        response_data = []
        for response in responses:
            response_data.append({
                'id': response.id,
                'score': response.score,
                'start_offset': response.start,
                'end_offset': response.end,
            })
        return response_data

    def _predict(self, requests):
        """
        Core function. Predict a list of requests using the loaded model, into
        a list of responses. Following steps are taken :
        * Convert each requests into a SQuAD-like example
        * Convert the SQuAD-like examples into Features
        * Create the `input_fn()` function from the Features
        * Predict (using the `input_fn()` function) raw results
        * Extract response's information from raw results

        Args:
            requests (list of Request): List of requests to answer. 

        Returns:
            list of Response: List of response, corresponding to each request
                given.
            
        """
        squad_items = convert_to_squad_example(requests)

        pred_features = []
        def append_feature(feature):
            pred_features.append(feature)

        convert_examples_to_features(
            examples=squad_items,
            tokenizer=self.tokenizer,
            max_seq_length=self.passage_max_len,
            doc_stride=self.doc_stride,
            max_query_length=self.question_max_len,
            is_training=False,
            output_fn=append_feature)

        pred_input_fn = input_fn_builder_no_file(
            records=pred_features,
            seq_length=self.passage_max_len,
            drop_remainder=False,
            batch_size=self.batch_size)

        raw_results = []
        for result in self.estimator.predict(pred_input_fn, 
                                             yield_single_examples=True):
            raw_results.append(RawResult(
                unique_id=int(result["unique_ids"]),
                start_logits=[float(x) for x in result["start_logits"].flat],
                end_logits=[float(x) for x in result["end_logits"].flat], 
                is_ans=[float(x) for x in result["is_ans"].flat],
                is_ans_max=float(result["is_ans_max"]),
                has_ans_max=float(result["has_ans_max"]),
                start_logits_sig=[float(x) for x in result["start_logits_sig"].flat],
                end_logits_sig=[float(x) for x in result["end_logits_sig"].flat]))

        return convert_to_response(
            all_examples=squad_items, 
            all_features=pred_features, 
            all_results=raw_results, 
            n_best_size=self.n_best,
            max_answer_length=self.answer_max_len, 
            do_lower_case=self.case)


######################## CLASSES FOR INTERFACE ################################

class Request():
    """
    Class used to represent a request to the interface. A request is someone 
    asking for an inference of a passage + a question.

    Attributes:
        id (str): ID of the request, uniquely identifying a request.
        title (str): Title of the passage. Not used in the interface currently.
        passage (str): Full passage to use for inference.
        question (str): Question to infer from the passage.
    """
    def __init__(self, id, title, passage, query):
        self.id = id
        self.title = title      # Not used
        self.passage = passage
        self.question = query

class Response():
    """
    Class used as the corresponding result of a request. A response contain the 
    answer to the corresponding request. The answer is represented with a
    character-level position for start and end of the answer, based on the
    original text of the passage.

    Attributes:
        id (str): ID of the request, uniquely identifying a request (and 
            therefore a response).
        score (float): Score of the response. Near 0 indicate a low confidence
            in the answer, near 1 indicate a high confidence in the answer.
        start (int): Character-level position of the beginning of the answer,
            based on the original text of the passage.
        end (int): Character-level position of the ending of the answer,
            based on the original text of the passage.
    """
    def __init__(self, id, score, start, end):
        self.id = id
        self.score = score
        self.start = start
        self.end = end


####################### FUNCTIONS FOR INTERFACE ###############################

def convert_to_squad_example(requests):
    """ 
    Equivalent of `read_squad_examples()`. Data is given as parameter of the
    function, and not as a file like previously.

    Note:
        Inspired from `inference.py`.

    Args: 
        requests (list of Request): Examples under the format of Request object.

    Returns:
        list of SquadExample: A list containing the examples summarized in 
            `SquadExample` object.
    
    """
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for req in requests:
        passage = req.passage
        question = req.question
        id = req.id

        doc_tokens = []
        prev_is_whitespace = True
        for c in passage:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False

        examples.append(SquadExample(
            qas_id=id,
            question_text=question,
            doc_tokens=doc_tokens,
            orig_answer_text=None,
            start_position=None,
            end_position=None,
            is_impossible=False))
    return examples


def input_fn_builder_no_file(records, seq_length, drop_remainder, batch_size):
    """ 
    Equivalent of `input_fn_builder()` but instead of writting records in a 
    file and using this file to construct the input, we simply pass the records
    to this function to build the input.

    Note:
        Taken from `inference.py`. A few changes done : removed useless 
        parameter and add a batch_size parameter.

    Args: 
        records (list of InputFeatures): The features to use as input.
        seq_length (int): Maximum sequence length of the input.
        drop_remainder (bool): Drop remainder (need better explanation).
        batch_size (int): The batch size to use.

    Returns:
        function: input_fn function, to be used in the estimator.
    
    """

    def input_fn(params):
        """The actual input function."""
        unique_ids = tf.expand_dims(tf.convert_to_tensor(records[0].unique_id, dtype=tf.int32), axis=0)
        input_ids = tf.expand_dims(tf.convert_to_tensor(records[0].input_ids, dtype=tf.int32), axis=0)
        input_mask = tf.expand_dims(tf.convert_to_tensor(records[0].input_mask, dtype=tf.int32), axis=0)
        segment_ids = tf.expand_dims(tf.convert_to_tensor(records[0].segment_ids, dtype=tf.int32), axis=0)
        s_mask = tf.expand_dims(tf.convert_to_tensor(records[0].s_mask, dtype=tf.int32), axis=0)
        e_mask = tf.expand_dims(tf.convert_to_tensor(records[0].e_mask, dtype=tf.int32), axis=0)
        q_mask = tf.expand_dims(tf.convert_to_tensor(records[0].q_mask, dtype=tf.int32), axis=0)
        for i in range(1, len(records)):
            unique_ids = tf.concat(
                [unique_ids, tf.expand_dims(tf.convert_to_tensor(records[i].unique_id, dtype=tf.int32), axis=0)],
                axis=0)
            input_ids = tf.concat(
                [input_ids, tf.expand_dims(tf.convert_to_tensor(records[i].input_ids, dtype=tf.int32), axis=0)],
                axis=0)
            input_mask = tf.concat(
                [input_mask, tf.expand_dims(tf.convert_to_tensor(records[i].input_mask, dtype=tf.int32), axis=0)],
                axis=0)
            segment_ids = tf.concat(
                [segment_ids, tf.expand_dims(tf.convert_to_tensor(records[i].segment_ids, dtype=tf.int32), axis=0)],
                axis=0)
            s_mask = tf.concat(
                [s_mask, tf.expand_dims(tf.convert_to_tensor(records[i].s_mask, dtype=tf.int32), axis=0)], axis=0)
            e_mask = tf.concat(
                [e_mask, tf.expand_dims(tf.convert_to_tensor(records[i].e_mask, dtype=tf.int32), axis=0)], axis=0)
            q_mask = tf.concat(
                [q_mask, tf.expand_dims(tf.convert_to_tensor(records[i].q_mask, dtype=tf.int32), axis=0)], axis=0)

        d = tf.data.Dataset.from_tensor_slices((unique_ids, input_ids, input_mask, segment_ids, s_mask, e_mask, q_mask))

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda unique_ids, input_ids, input_mask, segment_ids, s_mask, e_mask, q_mask:
                {'unique_ids': unique_ids, 'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
                 's_mask': s_mask, 'e_mask': e_mask, 'q_mask': q_mask},
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

def convert_to_response(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case):
    """ 
    Equivalent of `write_predictions()` but instead of writting predictions in 
    a file, this function convert the raw-results to responses.

    Note:
        Taken from `inference.py`.

    Args: 
        all_examples (list of SquadExample): The items to predict in a SQuAD
            format.
        all_features (list of InputFeatures): Features corresponding to the 
            SQuAD items.
        all_results (list of RawResult): Raw results corresponding to the SQuAD
            items.
        n_best_size (int): Number of n-best predictions to generate (used for 
            evaluation of SQuAD).
        max_answer_length (int): Maximum size of the answer, in token number.
        do_lower_case (bool): Is the input text lowercased ?

    Returns:
        list of Response: Answers to the questions, as a list of Response 
            objects.

    TODO : 
        Clean the function of all useless details. It is kept until now in case
        of changement of specifications, in order to not have to redo all the 
        details.
    
    """
    responses = []

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "is_ans", "is_ans_max", "has_ans_max"])

    all_predictions = collections.OrderedDict()
    all_predictions_pb = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits + result.start_logits_sig, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits + result.end_logits_sig, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            is_ans= sum(result.is_ans[start_index:end_index+1]) / (end_index+1-start_index) + result.start_logits_sig[start_index] + result.end_logits_sig[end_index],
                            is_ans_max = result.is_ans_max,
                            has_ans_max = result.has_ans_max
                            #is_ans= 1 / (1 + math.exp(-(sum(result.is_ans[start_index:end_index+1]) / (end_index+1-start_index))))
                            )
                        )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit + x.is_ans),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "is_ans", 
                                "is_ans_max", "has_ans_max", 
                                "char_index_start", "char_index_end"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            char_index_start, char_index_end = _get_independant_char_index(
                full_orig_tokens=example.doc_tokens, 
                start=orig_doc_start, 
                final_text=final_text
            )

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    is_ans=pred.is_ans, 
                    is_ans_max=pred.is_ans_max,
                    has_ans_max=pred.has_ans_max,
                    char_index_start=char_index_start,
                    char_index_end=char_index_end))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, 
                                 is_ans=-10.0, is_ans_max=-10.0, 
                                 has_ans_max=-10.0, char_index_start=-1,
                                 char_index_end=-1))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit + entry.is_ans)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["is_ans"] = entry.is_ans 
            output["is_ans_max"] = entry.is_ans_max
            output["has_ans_max"] = entry.has_ans_max
            output["char_index_start"] = entry.char_index_start
            output["char_index_end"] = entry.char_index_end
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if -(nbest_json[0]["is_ans"] + nbest_json[0]["is_ans_max"] + nbest_json[0]["has_ans_max"]) > -5.224823812643687:
            start_offset = -1
            end_offset = -1
        else:
            start_offset = nbest_json[0]["char_index_start"]
            end_offset = nbest_json[0]["char_index_end"]

        # Create the response
        responses.append(Response(
            id=example.qas_id,
            score=nbest_json[0]["probability"],
            start=start_offset,
            end=end_offset))
    return responses

def _get_independant_char_index(full_orig_tokens, start, final_text):
    """
    Function used to extract the character-based position of the prediction.
    Indeed the original prediction gives is a token-based position, related to 
    the tokenized version of the passage. But we want something independant of
    the tokenized passage. So in order to do that, we extract the 
    character-based position of the answer on the original passage (not 
    tokenized).

    Args:
        full_orig_tokens (list of str): List of original tokens (not tokenized).
        start (int): Index of start position for the predicted answer.
        final_text (str): Final version of the answer, based on the tokenized
            version of the answer (see the function `get_final_text()`).

    Returns:
        int: Character-based start position of the prediction for the original
            passage (not tokenized).
        int: Character-based start position of the prediction for the original
            passage (not tokenized).
            
    """
    # First, count the number of characters in the token before start_index
    char_nb = 0
    for i in range(start):
        char_nb += len(full_orig_tokens[i]) + 1
        #                                   + 1 : Because there is space 
        #                                         between each words 
    
    # So we have the character-based position of the start position
    char_pos_start = char_nb

    # To compute the end position, we use the final_text, in order to omit 
    # useless character ('s) removed by `get_final_text()`
    char_pos_end = char_pos_start + len(final_text)

    return char_pos_start, char_pos_end