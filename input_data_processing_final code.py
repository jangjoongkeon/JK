#Generating input files for pre-training data input

'''
for generating input files for pre-training, i make following code

logic of this code  is as follows

first step, Read wiki_corpus

second step, Divide the paragraph in the corpus by "." using split fuction

third step, Consider number relate to "." ex) 5.18, 34.8% ... etc

final step, Write sentence divided from paragraph

'''

for i in range(1,11):
    with open("/data01/Data/wiki_kr/위키피디아_말뭉치_{}.txt".format(i), "r") as fp: #input_data directory
        wiki_corpus = fp.readlines()
        print("{}번째 파일을 처리했습니다.".format(i-1))
        with open("/data01/Data/output/위키피디아_말뭉치_처리완료_{}.txt".format(i), "w") as w:
            for paragraph in wiki_corpus:
                if '\"' in paragraph:
                    w.write(paragraph)
                    continue
                sentences = paragraph.split(".")
                prev_str = None
                for token in sentences:
                    if len(token) == 0:  # for ... part,
                        continue
                    if prev_str is not None:
                        if len(token) != 0 and token[0].isdigit():  # if first value of token is int?
                            token = prev_str + "." + token
                        else:
                            prev_str = prev_str.lstrip()
                            w.write(prev_str + "." + "\n")
                        prev_str = None
                    if len(token) != 0 and token[-1].isdigit():  # if end value of token is int?
                        prev_str = token
                    else:
                        if token == '\n' or token ==' \n':    #prevent making '.' as applying '\n'
                            w.write(token)
                        else:
                            token = token.lstrip()
                            w.write(token + "." + "\n")
                if prev_str is not None:
                    prev_str = prev_str.lstrip()
                    w.write(prev_str + '.' + '\n')
