import argparse
import unittest
import json
import os
from jsonschema import validate

from en.interface import *

TEST_CONFIG_FILE = 'en/unit_test_config.json'
RESPONSE_SCHEMA = {
    'type': 'array',
    'additionalItems': False,
    'minItems': 1,
    'uniqueItems': True,
    'items': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'id': {'type': 'string', 'minLength': 1},
            'score': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'start_offset': {'type': 'integer', 'minimum': -1},
            'end_offset': {'type': 'integer', 'minimum': -1}
        },
        'required': ['id', 'score', 'start_offset', 'end_offset']
    }
}

class TestInterface(unittest.TestCase):
    """ Tests for the Interface. """

    @classmethod
    def setUpClass(cls):
        """ Create just 1 interface for all tests, stored as class variable. """
        super(TestInterface, cls).setUpClass()

        # Retrieve test configuration from configuration file
        with open(TEST_CONFIG_FILE) as f:
            cls.config = json.load(f)

        # Setup GPU 
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.config['gpu']

        # Create the interface
        cls.interface = SquadInference(
            config_file=cls.config['bert_config_file'],
            vocab_file=cls.config['bert_vocab_file'],
            lower_case=cls.config['lower_case'],
            tmp_dir=cls.config['tmp_dir'],
            model_ckpt=cls.config['bert_checkpoint_file'],
            lr=3e-5,
            passage_max_len=cls.config['context_max_len'],
            doc_stride=cls.config['doc_stride'],
            question_max_len=cls.config['question_max_len'],
            batch_size=cls.config['batch_size'],
            n_best=cls.config['n_best'],
            answer_max_len=cls.config['answer_max_len'],
        )

        print("\t\t\t===> Interface created <===")

    ######################### GENERAL TESTS ###################################

    def test_response_format(self):
        request = {
            'query': 'What is your name ?',
            'passages': [
                {
                    'id': '0a',
                    'title': 'test',
                    'content': 'My name is Nicolas.'
                }
            ]
        }

        response = self.interface.predict_request(request)

        try:
            validate(response, RESPONSE_SCHEMA)
        except:
            self.fail("\n\t\tWrong response format. Response : {}".format(response))

    def test_more_request_than_batch_size(self):
        # Send 2.5 * batch_size requests. If batch_size = 8 => send 20 requests
        bs = self.config['batch_size']
        request_nb = (bs * 2) + bs // 2
        
        passages = []
        for i in range(request_nb):
            passages.append({
                'id': str(i), 
                'title': "test{}".format(i),
                'content': "My name is Nicolas{}".format(i),
            })

        request = {
            'query': 'What is your name ?',
            'passages': passages
        }

        response = self.interface.predict_request(request)

        try:
            validate(response, RESPONSE_SCHEMA)
        except:
            self.fail("\n\t\tWrong response format for several requests. "
                      "Response : {}".format(response))
        self.assertEqual(request_nb, len(response), "\n\t\tSent {} requests but"
                         "received {} responses.".format(request_nb, len(response)))


    ######################## ANSWERABLE TESTS #################################

    def test_predict_answerable_synonymy(self):   
        _test_predict_item(
            interface=self.interface, 
            p='The Rankine cycle is sometimes referred to as a practical Carnot cycle.', 
            q='What is the Rankine cycle sometimes called ?', 
            a='practical Carnot cycle', 
            assertion=self.assertTrue)
    
    def test_predict_answerable_world_knowledge(self):   
        _test_predict_item(
            interface=self.interface, 
            p='The European Parliament and the Council of the European Union have powers of amendment and veto during the legislative process.', 
            q='Which governing bodies have veto power ?', 
            a='The European Parliament and the Council of the European Union', 
            assertion=self.assertTrue)

    def test_predict_answerable_syntactic_variation(self):   
        _test_predict_item(
            interface=self.interface, 
            p='Current faculty include the anthropologist Marshall Sahlins, Shakespeare scholar David Bevington.', 
            q='What Shakespeare scholar is currently on the faculty ?', 
            a='David Bevington', 
            assertion=self.assertTrue)

    def test_predict_answerable_multiple_sentence_reasoning(self):   
        _test_predict_item(
            interface=self.interface, 
            p='The V&A Theatre & Performance galleries opened in March 2009. They hold the UK’s biggest national collection of material about live performance.', 
            q='What collection does the V&A Theatre & Performance galleries hold ?', 
            a='material about live performance', 
            assertion=self.assertTrue)

    def test_predict_answerable_ambiguous(self):   
        _test_predict_item(
            interface=self.interface, 
            p='Achieving crime control via incapacitation and deterrence is a major goal of criminal punishment.', 
            q='What is the main goal of criminal punishment ?', 
            a='incapacitation', 
            assertion=self.assertTrue)

    def test_predict_answerable_bigger_than_384(self):
        _test_predict_item(
            interface=self.interface,
            p="Both X.25 and Frame Relay provide connection-oriented operations. But X.25 does it at the network layer of the OSI Model. Frame Relay does it at level two, the data link layer. Another major difference between X.25 and Frame Relay is that X.25 requires a handshake between the communicating parties before any user packets are transmitted. Frame Relay does not define any such handshakes. X.25 does not define any operations inside the packet network. It only operates at the user-network-interface (UNI). Thus, the network provider is free to use any procedure it wishes inside the network. X.25 does specify some limited re-transmission procedures at the UNI, and its link layer protocol (LAPB) provides conventional HDLC-type link management procedures. Frame Relay is a modified version of ISDN's layer two protocol, LAPD and LAPB. As such, its integrity operations pertain only between nodes on a link, not end-to-end. Any retransmissions must be carried out by higher layer protocols. The X.25 UNI protocol is part of the X.25 protocol suite, which consists of the lower three layers of the OSI Model. It was widely used at the UNI for packet switching networks during the 1980s and early 1990s, to provide a standardized interface into and out of packet networks. Some implementations used X.25 within the network as well, but its connection-oriented features made this setup cumbersome and inefficient. Frame relay operates principally at layer two of the OSI Model. However, its address field (the Data Link Connection ID, or DLCI) can be used at the OSI network layer, with a minimum set of procedures. Thus, it rids itself of many X.25 layer 3 encumbrances, but still has the DLCI as an ID beyond a node-to-node layer two link protocol. The simplicity of Frame Relay makes it faster and more efficient than X.25. Because Frame relay is a data link layer protocol, like X.25 it does not define internal network routing operations. For X.25 its packet IDs---the virtual circuit and virtual channel numbers have to be correlated to network addresses. The same is true for Frame Relays DLCI. How this is done is up to the network provider. Frame Relay, by virtue of having no network layer procedures is connection-oriented at layer two, by using the HDLC/LAPD/LAPB Set Asynchronous Balanced Mode (SABM). X.25 connections are typically established for each communication session, but it does have a feature allowing a limited amount of traffic to be passed across the UNI without the connection-oriented handshake. For a while, Frame Relay was used to interconnect LANs across wide area networks. However, X.25 and well as Frame Relay have been supplanted by the Internet Protocol (IP) at the network layer, and the Asynchronous Transfer Mode (ATM) and or versions of Multi-Protocol Label Switching (MPLS) at layer two. A typical configuration is to run IP over ATM or a version of MPLS. <Uyless Black, X.25 and Related Protocols, IEEE Computer Society, 1991> <Uyless Black, Frame Relay Networks, McGraw-Hill, 1998> <Uyless Black, MPLS and Label Switching Networks, Prentice Hall, 2001> < Uyless Black, ATM, Volume I, Prentice Hall, 1995>",
            q='What is a typical configuration',
            a='A typical configuration is to run IP over ATM or a version of MPLS',
            assertion=self.assertTrue)


    ###################### NON-ANSWERABLE TESTS ###############################

    def test_predict_non_answerable_negation(self):   
        _test_predict_item(
            interface=self.interface, 
            p='Several hospital pharmacies have decided to outsource high risk preparations.', 
            q='What types of pharmacy functions have never been outsourced ?', 
            a=None, 
            assertion=self.assertTrue)

    def test_predict_non_answerable_antonym(self):   
        _test_predict_item(
            interface=self.interface, 
            p='the extinction of the dinosaurs allowed the tropical rainforest to spread out across the continent.', 
            q='The extinction of what led to the decline of rainforests ?', 
            a=None, 
            assertion=self.assertTrue)

    def test_predict_non_answerable_entity_swap(self):   
        _test_predict_item(
            interface=self.interface, 
            p='These values are much greater than the 9–88 cm as projected in its Third Assessment Report.', 
            q='What was the projection of sea level increases in the fourth assessment report ?', 
            a=None, 
            assertion=self.assertTrue)

    def test_predict_non_answerable_mutual_exclusion(self):   
        _test_predict_item(
            interface=self.interface, 
            p='BSkyB waived the charge for subscribers whose package included two or more premium channels.', 
            q='What service did BSkyB give away for free unconditionally ?', 
            a=None, 
            assertion=self.assertTrue)

    def test_predict_non_answerable_impossible_condition(self):   
        _test_predict_item(
            interface=self.interface, 
            p='Union forces left Jacksonville and confronted a Confederate Army at the Battle of Olustee. Union forces then retreated to Jacksonville and held the city for the remainder of the war.', 
            q='After what battle did Union forces leave Jacksonville for good ?', 
            a=None, 
            assertion=self.assertTrue)

    def test_predict_non_answerable_other_neutral(self):   
        _test_predict_item(
            interface=self.interface, 
            p='Schuenemann et al. concluded in 2011 that the Black Death was caused by a variant of Y. pestis.', 
            q='Who discovered Y. pestis ?', 
            a=None, 
            assertion=self.assertTrue)

    ########################## APOSTROPHE TESTS ###############################

    def test_predict_answer_with_apostrophe(self):   
        _test_predict_item(
            interface=self.interface, 
            p="John Smith's daughter went to the store.", 
            q="Who's daughter went to the store ?", 
            a="John Smith's", 
            assertion=self.assertTrue)

    def test_predict_answer_without_apostrophe(self):   
        _test_predict_item(
            interface=self.interface, 
            p="John Smith's daughter went to the store.", 
            q="What is the name of the father of the daughter who went to the store ?", 
            a="John Smith", 
            assertion=self.assertTrue)


def _test_predict_item(interface, p, q, a, assertion):
    """
    Function to test the interface given a passage, a question, and the
    supposedly right answer. This function simply predict the answer using the
    given interface and compare it with the true answer. Answer is tested 
    through an assertion.

    Args:
        interface (SquadInference): Interface to use for prediction.
        p (str): Passage.
        q (str): Question.
        a (str): True answer, or `None` if not answerable.
        assertion (function): `assertTrue` function from the TestCase.
    """
    request = { 
        'query': q, 
        'passages': [{'id': '0a', 'title': 'test', 'content': p}]
    }

    response = interface.predict_request(request)
    r = response[0]

    if a is None:
        a_s = -1
        a_e = -1
    else:
        a_s = p.index(a)
        a_e = a_s + len(a)
    assertion(a_s == r['start_offset'] and a_e == r['end_offset'], 
              "\n\tPassage : {}\n\n\tQuestion : {}\n\t\tAnswer : {} ({} - {})"
              "\n\t\tPrediction : {} ({} - {})\n".format(p, q, a, a_s, a_e, 
              p[r['start_offset']:r['end_offset']], r['start_offset'], 
              r['end_offset']))


if __name__ == '__main__':
    unittest.main()