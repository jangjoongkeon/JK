# coding=utf-8

import unittest

import bert.tokenization as tokenization

class TestFullTokenizer(unittest.TestCase):
    def test_tokenize(self):
        vocab_file = r"C:\MyProgramData\Pretrained Models\BERT\multi_cased_L-12_H-768_A-12\vocab.txt"
        # vocab_file = r"/home/wtjeong/workspace/Data/bert/multi_cased_L-12_H-768_A-12/vocab.txt"
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)

        input = "아버지는 대한민국을 사랑한다."
        tokens = tokenizer.tokenize(input, one_char_subword=True)
        print("input len: {}, tokens len:{}, {}".format(len(input), len(tokens), tokens))

if __name__ == "__main__":
    unittest.main()
