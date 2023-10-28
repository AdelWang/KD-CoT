import pandas as pd
import os
import json
from tqdm.auto import tqdm
import time
import re
from prompts import (
    INSTRUCTION,
    CWQ_EXAMPLE,
    WEQSP_EXAMPLE,
    INSTRUCTION_COT,
    CWQ_EXAMPLE_COT,
    INSTRUCTION_DIRECT,
    CWQ_EXAMPLE_DIRECT,
    WEBQSP_EXAMPLE_DIRECT,
    WEBQSP_EXAMPLE_COT
)
import random
import unicodedata
import jsonlines

def cleantxt(raw):
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.,"]+', re.UNICODE)
    raw = fil.sub('', raw)
    return raw.lower()

class GPT_Infenrence:
    def __init__(
        self,
        temp_dir_path : str,
        webqsp_path : str = None,
        cwq_path : str = None,
        iteration : int = 0,
        temp_file_name: str = 'temp_file',
        max_iteration = 3
    ):
        self.temp_dir_path = temp_dir_path
        self.webqsp_temp_file = os.path.join(self.temp_dir_path, 'webqsp_' + temp_file_name + '.json')
        self.cwq_temp_file = os.path.join(self.temp_dir_path, 'cwq_' + temp_file_name + '.json')
        self.iteration = iteration
        if not os.path.exists(self.webqsp_temp_file):
            self.load_webqsp(webqsp_path)
        if not os.path.exists(self.cwq_temp_file):
            self.load_cwq(cwq_path)
        self.max_iteration = max_iteration

    def update_iteration(self):
        self.iteration += 1

    ################################################################
    # WEBQSP
    ################################################################

    def load_webqsp(self, data_path : str):
        def get_answers(question):
            """extract unique answers from question parses."""
            answer_list = []
            for parse in question['Parses']:
                for answer in parse['Answers']:
                    entity = answer["EntityName"]
                    argument = answer["AnswerArgument"]
                    if argument.startswith("m.") or argument.startswith("g."):
                        argument = '/' + argument.replace('.', '/')
                    if entity:
                        answer_list.append(entity)
                    else:
                        answer_list.append(argument)
            answer_list = list(set(answer_list))
            answer_list = [answer for answer in answer_list if answer]
            if len(answer_list) == 0:
                answer_list = ['']
            return answer_list

        data = json.load(open(data_path, encoding='UTF-8'))
        data = data['Questions']

        webqsp_questions = []
        for question in data:
            q_obj = {
                "id": question["QuestionId"],
                "question": question["RawQuestion"][0].upper() + question["RawQuestion"][1:],
                "answers": get_answers(question),
                "PredictedAnswer": -1,
                "IntermediateAction": [],
                "IntermediateAnswer": [],
                "IntermediateThought": [],
                "IntermediateGPTAnswer": [],
                "IntermediateFiDAnswer": []
            }
            if "demonstration" in question:
                q_obj['demon'] = question['demonstration']
            webqsp_questions.append(q_obj)
        json.dump(webqsp_questions, open(self.webqsp_temp_file, 'w'), indent=2)
    
    def construct_webqsp_prompts(self, format : str = 'gpt3.5', special_demon : bool = False, version_num = 5, demon=True, think = False):
        original_prompts = INSTRUCTION
        data = json.load(open(self.webqsp_temp_file))
        webqsp_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                continue
            
            if demon:
                original_prompts = INSTRUCTION + WEQSP_EXAMPLE

            if special_demon:
                original_prompts = INSTRUCTION + question['demon']

            prompt = original_prompts + '\nQuestion ' + question['question'] + '\n' 

            if think:
                prompt += " Let's think step by step: \n"

            if self.iteration != 0:
                for j in range(self.iteration):
                    prompt += 'Thought {} '.format(j + 1) + question['IntermediateThought'][j] + '\n'
                    prompt += 'Action {} '.format(j + 1) + question['IntermediateAction'][j][0] + '[' + question['IntermediateAction'][j][1] + ']\n'
                    prompt += 'Answer {} '.format(j + 1) + question['IntermediateAnswer'][j] + '\n'
            prompt += 'Thought {} '.format(self.iteration + 1) 
            webqsp_prompts.append({'id': question['id'], 'input': prompt, 'target': 'unk'})
        
        print('Webqsp prompts number:', len(webqsp_prompts))
        results = [webqsp_prompts[i:min(i+1000, len(webqsp_prompts))] for i in range(0, len(webqsp_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'webqsp_prompts{}'.format(version_num))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '.json')
            json.dump(webqsp_prompts, open(out_path, 'w'), indent=2)

            out_path = os.path.join(out_dir, 'train.json')
            with jsonlines.open(out_path, 'w') as writer:
                for prompt in webqsp_prompts:
                    writer.write(prompt)

            out_path = os.path.join(out_dir, 'dev.json')
            with jsonlines.open(out_path, 'w') as writer:
                for prompt in webqsp_prompts:
                    writer.write(prompt)
    
    def construct_webqsp_prompts_cot(self, format : str = 'gpt3.5'):
        original_prompts = INSTRUCTION_COT + WEBQSP_EXAMPLE_COT
        data = json.load(open(self.webqsp_temp_file))
        webqsp_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                print(question['PredictedAnswer'])
                continue

            prompt = original_prompts + '\nQ: ' + question['question'] + '\n' 
            prompt += 'A: '
            webqsp_prompts.append({'id': question['id'], 'input': prompt})
        results = [webqsp_prompts[i:min(i+1000, len(webqsp_prompts))] for i in range(0, len(webqsp_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'webqsp_prompts_cot')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '.json')
            json.dump(webqsp_prompts, open(out_path, 'w'), indent=2)
    
    def construct_webqsp_prompts_direct(self, format : str = 'gpt3.5', special_demon : bool = False):
        original_prompts = INSTRUCTION_DIRECT + WEBQSP_EXAMPLE_DIRECT
        data = json.load(open(self.webqsp_temp_file))
        webqsp_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                continue

            prompt = original_prompts + '\nQ: ' + question['question'] + '\n' 
            prompt += 'A: '
            webqsp_prompts.append({'id': question['id'], 'input': prompt})
        results = [webqsp_prompts[i:min(i+1000, len(webqsp_prompts))] for i in range(0, len(webqsp_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'webqsp_prompts1')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'webqsp_prompts' + str(self.iteration) + '.json')
            json.dump(webqsp_prompts, open(out_path, 'w'), indent=2)
    
    def process_webqsp_response(self, version_num = 5, think=False):
        temp_data = json.load(open(self.webqsp_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'webqsp_outputs' + str(version_num) + '/webqsp_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            if not think:
                gpt_response = pd.read_csv(response_path, encoding='gb18030') 
            else:
                gpt_response = pd.read_csv(response_path, encoding='utf-8') 
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQuestion ')
                key_e = gpt_response['origin_question'][j][key_s:].find('\nThought 1') + key_s
                key = gpt_response['origin_question'][j][key_s:key_e]

                if think:
                    key = gpt_response['origin_question'][j][key_s:]
                    key = key.split("Let's think step by step:")[0]

                key = cleantxt(key.replace('\nQuestion ', '').strip().rstrip())
                assert key in temp_data_dic, gpt_response['origin_question'][j] + '\n' + key
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})
        
        def get_answer(text):
            answer = text.split('Finish[')[-1]
            answer = answer.strip().rstrip()
            if len(answer) == 0:
                return answer
            if ']' == answer[-1]:
                answer = answer[:-1]
            if '.' == answer[-1]:
                answer = answer[:-1]
            return answer

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            tp = 0

            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    # print(gold, '|||', pre)
                    tp += 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            tp += 1
            precision = int(tp > 0)
            recall = tp / len(gold_list)
            f1 = (2 * precision * recall) / (precision + recall + 1e-40)
            return precision, f1

        total = 0
        sumc = 0
        sumf1 = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            answer = answer.lower().strip().rstrip()
            gold_list = [gold.lower().strip() for gold in gold_list]
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            hit, f1 = exact_match(answer, pre_proc_gold_list, gold_list, question)
            sumc += hit
            sumf1 += f1
        print("WEBSP EM: {} / {} = {:.3f} | F1: {:.3f} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total, sumf1, total, sumf1 * 1.0 / total))
        ####################
        
        def action_normalization(action):
            import editdistance

            action_list = ['Question', 'Multi_Answer_Question', 'Finish']
            max_score = -100000
            max_action = -1
            for a in action_list:
                score = -editdistance.distance(a, action)
                if score > max_score:
                    max_score = score
                    max_action = a
            return max_action
        
        def question_normalization(text):
            text = str(text)
            stop_token = ['\n', '\t', '?', '!']
            for t in stop_token:
                text = text.split(t)[0]
            if re.search(r'Thought( |\n)*[0-9]', text):
                text = re.search(r'.*Thought( |\n)*[0-9]', text).group()
                temp = re.search(r'Thought( |\n)*[0-9]', text).group()
                text = text.replace(temp, '')
            if len(text) == 0:
                text = 'unk'
            return text

        for result in result_list:
            key = result['question']
            output = result['output']
            if 'Thought {}'.format(self.iteration + 2) not in output:
                answer = get_answer(output)
                temp_data_dic[key]['PredictedAnswer'] = answer
            elif self.iteration >= self.max_iteration:
                answer = get_answer(output)
                temp_data_dic[key]['PredictedAnswer'] = answer
            else:
                try:
                    thought = re.search(r'(.|\n)*?(Action|Event) ' + str(self.iteration + 1), output).group()
                    thought = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', thought).strip().rstrip()

                    if 'Action ' + str(self.iteration + 1) + '\n' in output:
                        output = output.replace('Action ' + str(self.iteration + 1) + '\n', 'Action ' + str(self.iteration + 1))

                    action = re.search(r'(Action|Event) ' + str(self.iteration + 1) + r'\n*.*?(]|\n)', output).group()
                    action = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', action).replace(']', '').strip().rstrip()
                    if len(action) < 3:
                        action_s = output.find('Action ' + str(self.iteration + 1))
                        output = output[action_s:].replace('Action ' + str(self.iteration + 1), '').strip().rstrip()
                        output = 'Action ' + str(self.iteration + 1) + ' ' + output
                        action = re.search(r'(Action|Event) ' + str(self.iteration + 1) + r'\n*.*?(]|\n)', output).group()
                        action = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', action).replace(']', '').strip().rstrip()
                    if '[' in action:
                        action = action.split('[')
                    else:
                        action = action.split(' ')
                    if len(action) != 2:
                        answer = get_answer(output)
                        temp_data_dic[key]['PredictedAnswer'] = answer
                        continue

                    action[0] = action_normalization(action[0])
                    action[1] = action[1]
                    if action[0] in ['Question', 'Multi_Answer_Question']:
                        action[1] = question_normalization(action[1])
                        if action[1] == 'unk':
                            answer = get_answer(output)
                            temp_data_dic[key]['PredictedAnswer'] = answer
                            continue
                    
                    gpt_answer = re.search(r'Answer ' + str(self.iteration + 1) + r'(.|\n)*?(Thought|Action|Answer)', output).group()
                    gpt_answer = gpt_answer.replace('Answer ' + str(self.iteration + 1), '').replace('Thought', '').strip().rstrip()
                    if 'Action' == gpt_answer[-6:]:
                        gpt_answer = gpt_answer[:-6]
                    if 'Answer' == gpt_answer[-6:]:
                        gpt_answer = gpt_answer[:-6]
                    gpt_answer = gpt_answer.replace('\n', ',')
                    if '.' == gpt_answer[-1]:
                        gpt_answer = gpt_answer[:-1]
                    if len(gpt_answer) > 300:
                        gpt_answer = re.split(r',|.|?|!|\n', gpt_answer)
                        gpt_answer_list = [t.strip().rstrip() for t in gpt_answer]
                        gpt_answer = ''
                        for t in gpt_answer_list:
                            if len(gpt_answer) + len(t) + 2 < 300:
                                gpt_answer += ', ' + t
                    assert len(gpt_answer) <= 300, 'GPT answer too long !!!'
                except:
                    answer = str(result['output']).split('Finish[')[-1]
                    answer = answer.strip().rstrip()
                    if len(answer) == 0:
                        return answer
                    if ']' == answer[-1]:
                        answer = answer[:-1]
                    if '.' == answer[-1]:
                        answer = answer[:-1]
                    temp_data_dic[key]['PredictedAnswer'] = answer
                    continue
                if action[0] == 'Finish':
                    temp_data_dic[key]['PredictedAnswer'] = action[1]
                else:
                    temp_data_dic[key]['IntermediateThought'].append(thought)
                    temp_data_dic[key]['IntermediateAction'].append(action)
                    temp_data_dic[key]['IntermediateGPTAnswer'].append(gpt_answer)
        new_temp_data = [v for k, v in temp_data_dic.items()]
        json.dump(new_temp_data, open(self.webqsp_temp_file, 'w'), indent=2)

    def webqsp_eval(self):
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            tp = 0

            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    # print(gold, '|||', pre)
                    tp += 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            tp += 1
            precision = int(tp > 0)
            recall = tp / len(gold_list)
            f1 = (2 * precision * recall) / (precision + recall + 1e-40)
            return precision, f1

        temp_data = json.load(open(self.webqsp_temp_file))
        sumc = 0
        sumf1 = 0
        for data in temp_data:
            question = data['question']
            answer = str(data['PredictedAnswer'])
            gold_list = data['answers']

            answer = answer.lower().strip().rstrip()
            gold_list = [gold.lower().strip() for gold in gold_list]
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = question.lower().strip()

            hit, f1 = exact_match(answer, pre_proc_gold_list, gold_list, question)
            sumc += hit
            sumf1 += f1
        print("WEBSP EM: {} / {} = {:.3f} | F1: {:.3f} / {} = {:.3f}".format(sumc, len(temp_data), sumc * 1.0 / len(temp_data), sumf1, len(temp_data), sumf1 * 1.0 / len(temp_data)))
    
    def process_webqsp_response_cot(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        # num_file = 2
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'webqsp_prompts_cot/webqsp_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            gpt_response = pd.read_csv(response_path, encoding='gb18030') 
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQ: ')
                key_e = gpt_response['origin_question'][j][key_s:].find('A:') + key_s
                key = cleantxt(gpt_response['origin_question'][j][key_s:key_e].replace('\nQ: ', '').strip().rstrip())
                assert key in temp_data_dic, gpt_response['origin_question'][j]
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})

        def get_answer(text):
            answer_s = str(text).rfind('So the answer is ')
            if answer_s != -1:
                answer_e = str(text[answer_s:]).rfind('.') + answer_s
            if answer_s == -1 or answer_e == -1:
                answer = 'unk'
            else:
                answer = text.strip()[answer_s:answer_e].replace('So the answer is ', '')
                if '.' == answer[-1]:
                    answer = answer[:-1]
            return answer

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            tp = 0

            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    # print(gold, '|||', pre)
                    tp += 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            tp += 1
            precision = int(tp > 0)
            recall = tp / len(gold_list)
            f1 = (2 * precision * recall) / (precision + recall + 1e-40)
            return precision, f1

        total = 0
        sumc = 0
        sumf1 = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            hit, f1 = exact_match(answer, pre_proc_gold_list, gold_list, question)
            sumc += hit
            sumf1 += f1
        print("WEBSP EM: {} / {} = {:.3f} | F1: {:.3f} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total, sumf1, total, sumf1 * 1.0 / total))
        ####################
    
    def process_webqsp_response_direct(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        # num_file = 2
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'webqsp_outputs1/webqsp_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            gpt_response = pd.read_csv(response_path, encoding='gb18030') 
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQ: ')
                key_e = gpt_response['origin_question'][j][key_s:].find('A:') + key_s
                key = cleantxt(gpt_response['origin_question'][j][key_s:key_e].replace('\nQ: ', '').strip().rstrip())
                assert key in temp_data_dic, gpt_response['origin_question'][j]
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})

        def get_answer(text):
            answer = text
            if '.' == answer[-1]:
                answer = answer[:-1]
            return answer.strip().rstrip()

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            tp = 0

            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    # print(gold, '|||', pre)
                    tp += 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            tp += 1
            precision = int(tp > 0)
            recall = tp / len(gold_list)
            f1 = (2 * precision * recall) / (precision + recall + 1e-40)
            return precision, f1

        total = 0
        sumc = 0
        sumf1 = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            answer = answer.lower().strip().rstrip()
            gold_list = [gold.lower().strip() for gold in gold_list]
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            hit, f1 = exact_match(answer, pre_proc_gold_list, gold_list, question)
            sumc += hit
            sumf1 += f1
        print("WEBSP EM: {} / {} = {:.3f} | F1: {:.3f} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total, sumf1, total, sumf1 * 1.0 / total))
        ####################

    ################################################################
    # CWQ
    ################################################################

    def load_cwq(self, data_path : str):
        def get_answers(question):
            """extract unique answers from question parses."""
            answer_list = []
            for answers in question["answers"]:
                if answers['answer']:
                    answer_list.append(answers['answer'])
                else:
                    answer_list.append(answers['answer_id'])
                if 'aliases' in answers:
                    answer_list += answers['aliases']
                else:
                    answer_list += answers['alias']
            answer_list = list(set(answer_list))
            answer_list = [answer for answer in answer_list if answer]
            if len(answer_list) == 0:
                answer_list = ['']
            return answer_list

        def get_non_alias_answers(question):
            """extract unique answers from question parses."""
            answer_list = []
            for answers in question["answers"]:
                if answers['answer']:
                    answer_list.append(answers['answer'])
                else:
                    answer_list.append(answers['answer_id'])
            answer_list = list(set(answer_list))
            answer_list = [answer for answer in answer_list if answer]
            if len(answer_list) == 0:
                answer_list = ['']
            return answer_list 

        data = json.load(open(data_path))
        cwq_questions = []
        for question in data:
            q_obj = {
                "id": question["ID"],
                "question": question["question"],
                "answers": get_answers(question),
                "target": '\n'.join(get_non_alias_answers(question)),
                "PredictedAnswer": -1,
                "IntermediateAction": [],
                "IntermediateAnswer": [],
                "IntermediateThought": [],
                "IntermediateGPTAnswer": [],
                "IntermediateFiDAnswer": []
            }
            if "demonstration" in question:
                q_obj['demon'] = question['demonstration']
            cwq_questions.append(q_obj)
        json.dump(cwq_questions, open(self.cwq_temp_file, 'w'), indent=2)

    def construct_cwq_prompts(self, format : str = 'gpt3.5', special_demon : bool = False, version_num = 5, demon = True, think = False):
        original_prompts = INSTRUCTION
        data = json.load(open(self.cwq_temp_file))
        cwq_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                # print(question['PredictedAnswer'])
                continue
            
            if demon:
                original_prompts = INSTRUCTION + CWQ_EXAMPLE

            if special_demon:
                original_prompts = INSTRUCTION + question['demon']

            prompt = original_prompts + '\nQuestion ' + question['question'] + '\n' 

            if think:
                prompt += " Let's think step by step: \n"

            if self.iteration != 0:
                for j in range(self.iteration):
                    prompt += 'Thought {} '.format(j + 1) + question['IntermediateThought'][j] + '\n'
                    prompt += 'Action {} '.format(j + 1) + question['IntermediateAction'][j][0] + '[' + question['IntermediateAction'][j][1] + ']\n'
                    prompt += 'Answer {} '.format(j + 1) + question['IntermediateAnswer'][j] + '\n'
            prompt += 'Thought {} '.format(self.iteration + 1) 
            cwq_prompts.append({'id': question['id'], 'input': prompt, 'target': 'unk'})
        
        print('CWQ prompts number:', len(cwq_prompts))
        results = [cwq_prompts[i:min(i+1000, len(cwq_prompts))] for i in range(0, len(cwq_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'cwq_prompts{}'.format(version_num))
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '.json')
            json.dump(cwq_prompts, open(out_path, 'w'), indent=2)

            out_path = os.path.join(out_dir, 'train.json')
            with jsonlines.open(out_path, 'w') as writer:
                for prompt in cwq_prompts:
                    writer.write(prompt)

            out_path = os.path.join(out_dir, 'dev.json')
            with jsonlines.open(out_path, 'w') as writer:
                for prompt in cwq_prompts:
                    writer.write(prompt)

    def construct_cwq_prompts_cot(self, format : str = 'gpt3.5'):
        original_prompts = INSTRUCTION_COT + CWQ_EXAMPLE_COT
        data = json.load(open(self.cwq_temp_file))
        cwq_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                print(question['PredictedAnswer'])
                continue

            prompt = original_prompts + '\nQ: ' + question['question'] + '\n' 
            prompt += 'A: '
            cwq_prompts.append({'id': question['id'], 'input': prompt})
        results = [cwq_prompts[i:min(i+1000, len(cwq_prompts))] for i in range(0, len(cwq_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'cwq_prompts2')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '.json')
            json.dump(cwq_prompts, open(out_path, 'w'), indent=2)
    
    def construct_cwq_prompts_direct(self, format : str = 'gpt3.5', version_num=5):
        original_prompts = INSTRUCTION_DIRECT + CWQ_EXAMPLE_DIRECT
        data = json.load(open(self.cwq_temp_file))
        cwq_prompts = []
        for i, question in enumerate(data):
            if question['PredictedAnswer'] != -1:
                print(question['PredictedAnswer'])
                continue

            prompt = original_prompts + '\nQ: ' + question['question'] + '\n' 
            prompt += 'A: '
            cwq_prompts.append({'id': question['id'], 'input': prompt, 'target': 'unk'})
        results = [cwq_prompts[i:min(i+1000, len(cwq_prompts))] for i in range(0, len(cwq_prompts), 1000)]
        out_dir = os.path.join(self.temp_dir_path, 'cwq_prompts2')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if format == 'gpt3.5':
            for i, result in enumerate(results):
                out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '_' + str(i) + '.csv')
                result = [k['input'].replace('\n', '\t') for k in result]
                result = pd.Series(result)
                result.to_csv(out_path, header=None, index=None)
        else:
            out_path = os.path.join(out_dir, 'cwq_prompts' + str(self.iteration) + '.json')
            json.dump(cwq_prompts, open(out_path, 'w'), indent=2)
    
    def process_cwq_response(self, version_num = 5, think = False):
        temp_data = json.load(open(self.cwq_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        # num_file = 2
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'cwq_outputs' + str(version_num) + '/cwq_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            if not think:
                gpt_response = pd.read_csv(response_path, encoding='gb18030') 
                # gpt_response = pd.read_csv(response_path, encoding='utf-8')
            else:
                gpt_response = pd.read_csv(response_path, encoding='utf-8')
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQuestion ')
                key_e = gpt_response['origin_question'][j][key_s:].find('\nThought 1') + key_s
                key = gpt_response['origin_question'][j][key_s:key_e]
                # print(key_s, key_e)
                if think:
                    key = gpt_response['origin_question'][j][key_s:]
                    key = key.split("Let's think step by step:")[0]

                key = cleantxt(key.replace('\nQuestion ', '').strip().rstrip())
                if key == cleantxt("What is the capital of France?"):
                    key = cleantxt('In what movie does Logan Lerman play that was written by Darren Aronofsky?')
                assert key in temp_data_dic, gpt_response['origin_question'][j] + '\n' + key
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})

        def get_answer(text):
            answer = text.split('Finish[')[-1]
            answer = answer.strip().rstrip()
            if len(answer) == 0:
                return answer
            if ']' == answer[-1]:
                answer = answer[:-1]
            if len(answer) > 0 and '.' == answer[-1]:
                answer = answer[:-1]
            return answer

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    return 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            return 1
            return 0

        sumc = 0
        total = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            sumc += exact_match(answer, pre_proc_gold_list, gold_list, question)
        print("CWQ EM: {} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total))
        ####################
        
        def action_normalization(action):
            import editdistance

            action_list = ['Question', 'Multi_Answer_Question', 'Finish']
            max_score = -100000
            max_action = -1
            for a in action_list:
                score = -editdistance.distance(a, action)
                if score > max_score:
                    max_score = score
                    max_action = a
            return max_action
        
        def question_normalization(text):
            text = str(text)
            stop_token = ['\n', '\t', '?', '!']
            for t in stop_token:
                text = text.split(t)[0]
            if re.search(r'Thought( |\n)*[0-9]', text):
                text = re.search(r'.*Thought( |\n)*[0-9]', text).group()
                temp = re.search(r'Thought( |\n)*[0-9]', text).group()
                text = text.replace(temp, '')
            if len(text) == 0:
                text = 'unk'
            return text

        for result in result_list:
            key = result['question']
            output = result['output']
            if 'Thought {}'.format(self.iteration + 2) not in output:
                answer = get_answer(output)
                temp_data_dic[key]['PredictedAnswer'] = answer
            elif self.iteration >= self.max_iteration:
                answer = get_answer(output)
                temp_data_dic[key]['PredictedAnswer'] = answer
            else:
                try:
                    thought = re.search(r'(.|\n)*(Action|Event) ' + str(self.iteration + 1), output).group()
                    thought = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', thought).strip().rstrip()

                    if 'Action ' + str(self.iteration + 1) + '\n' in output:
                        output = output.replace('Action ' + str(self.iteration + 1) + '\n', 'Action ' + str(self.iteration + 1))

                    action = re.search(r'(Action|Event) ' + str(self.iteration + 1) + r'\n*.*?(]|\n)', output).group()
                    action = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', action).replace(']', '').strip().rstrip()
                    if len(action) < 3:
                        action_s = output.find('Action ' + str(self.iteration + 1))
                        output = output[action_s:].replace('Action ' + str(self.iteration + 1), '').strip().rstrip()
                        output = 'Action ' + str(self.iteration + 1) + ' ' + output
                        action = re.search(r'(Action|Event) ' + str(self.iteration + 1) + r'\n*.*?(]|\n)', output).group()
                        action = re.sub(r'(Action|Event) ' + str(self.iteration + 1), '', action).replace(']', '').strip().rstrip()
                    if '[' in action:
                        action = action.split('[')
                    else:
                        action = action.split(' ')
                    if len(action) != 2:
                        answer = get_answer(output)
                        temp_data_dic[key]['PredictedAnswer'] = answer
                        continue

                    action[0] = action_normalization(action[0])
                    action[1] = action[1]
                    if action[0] in ['Question', 'Multi_Answer_Question']:
                        action[1] = question_normalization(action[1])
                        if action[1] == 'unk':
                            answer = get_answer(output)
                            temp_data_dic[key]['PredictedAnswer'] = answer
                            continue
                    
                    gpt_answer = re.search(r'Answer ' + str(self.iteration + 1) + r'(.|\n)*?(Thought|Action|Answer)', output).group()
                    gpt_answer = gpt_answer.replace('Answer ' + str(self.iteration + 1), '').replace('Thought', '').strip().rstrip()
                    if 'Action' == gpt_answer[-6:]:
                        gpt_answer = gpt_answer[:-6]
                    if 'Answer' == gpt_answer[-6:]:
                        gpt_answer = gpt_answer[:-6]
                    gpt_answer = gpt_answer.replace('\n', ',')
                    if '.' == gpt_answer[-1]:
                        gpt_answer = gpt_answer[:-1]
                    if len(gpt_answer) > 300:
                        gpt_answer = re.split(r',|.|?|!|\n', gpt_answer)
                        gpt_answer_list = [t.strip().rstrip() for t in gpt_answer]
                        gpt_answer = ''
                        for t in gpt_answer_list:
                            if len(gpt_answer) + len(t) + 2 < 300:
                                gpt_answer += ', ' + t
                    assert len(gpt_answer) <= 300, 'GPT answer too long !!!'
                except:
                    answer = str(result['output']).split('Finish[')[-1]
                    answer = answer.strip().rstrip()
                    if len(answer) == 0:
                        return answer
                    if ']' == answer[-1]:
                        answer = answer[:-1]
                    if '.' == answer[-1]:
                        answer = answer[:-1]
                    temp_data_dic[key]['PredictedAnswer'] = answer
                    continue
                if action[0] == 'Finish':
                    temp_data_dic[key]['PredictedAnswer'] = action[1]
                else:
                    temp_data_dic[key]['IntermediateThought'].append(thought)
                    temp_data_dic[key]['IntermediateAction'].append(action)
                    temp_data_dic[key]['IntermediateGPTAnswer'].append(gpt_answer)
        new_temp_data = [v for k, v in temp_data_dic.items()]
        json.dump(new_temp_data, open(self.cwq_temp_file, 'w'), indent=2)

    def cwq_eval(self):
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    return 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            return 1
            return 0

        temp_data = json.load(open(self.cwq_temp_file))
        sumc = 0
        for data in temp_data:
            question = data['question']
            o = question
            answer = str(data['PredictedAnswer'])
            gold_list = data['answers']

            answer = answer.lower().strip().rstrip()
            gold_list = [gold.lower().strip() for gold in gold_list]
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = question.lower().strip()

            sumc += exact_match(answer, pre_proc_gold_list, gold_list, question)
        print("CWQ EM: {} / {} = {:.3f}".format(sumc, len(temp_data), sumc * 1.0 / len(temp_data)))
    
    def process_cwq_response_cot(self):
        temp_data = json.load(open(self.cwq_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'cwq_outputs2/cwq_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            gpt_response = pd.read_csv(response_path, encoding='gb18030') 
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQ: ')
                key_e = gpt_response['origin_question'][j][key_s:].find('A:') + key_s
                key = cleantxt(gpt_response['origin_question'][j][key_s:key_e].replace('\nQ: ', '').strip().rstrip())
                assert key in temp_data_dic, gpt_response['origin_question'][j]
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})

        def get_answer(text):
            answer_s = str(text).rfind('So the answer is ')
            if answer_s != -1:
                answer_e = str(text[answer_s:]).rfind('.') + answer_s
            if answer_s == -1 or answer_e == -1:
                answer = 'unk'
            else:
                answer = text.strip()[answer_s:answer_e].replace('So the answer is ', '')
                if '.' == answer[-1]:
                    answer = answer[:-1]
            return answer

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    print(gold, '|||', pre)
                    return 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            return 1
            return 0

        sumc = 0
        total = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            sumc += exact_match(answer, pre_proc_gold_list, gold_list, question)
        print("CWQ EM: {} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total))
        ####################
    
    def process_cwq_response_direct(self):
        temp_data = json.load(open(self.cwq_temp_file))
        temp_data_dic = {cleantxt(data['question']): data for data in temp_data}

        num_file = len(temp_data) // 1000
        if (len(temp_data) % 1000) != 0:
            num_file += 1
        # num_file = 2
        def process_text(text):
            return str(text).replace('\t', '\n')

        result_list = []
        for i in range(1, num_file+1):
            file = 'cwq_outputs2/cwq_output' + str(self.iteration) + '_' + str(i) + '.csv'
            response_path = os.path.join(self.temp_dir_path, file)
            gpt_response = pd.read_csv(response_path, encoding='gb18030') 
            gpt_response['origin_question'] = gpt_response['origin_question'].apply(process_text)
            gpt_response['origin_answer'] = gpt_response['origin_answer'].apply(process_text)

            for j in range(len(gpt_response)):
                key_s = gpt_response['origin_question'][j].rfind('\nQ: ')
                key_e = gpt_response['origin_question'][j][key_s:].find('A:') + key_s
                key = cleantxt(gpt_response['origin_question'][j][key_s:key_e].replace('\nQ: ', '').strip().rstrip())
                assert key in temp_data_dic, gpt_response['origin_question'][j]
                result_list.append({'question': key, 'output': gpt_response['origin_answer'][j]})

        def get_answer(text):
            answer = text
            if '.' == answer[-1]:
                answer = answer[:-1]
            return answer.strip().rstrip()

        ####################
        # 评测当前步骤
        def exact_match(pre, pre_proc_gold_list, gold_list, question):
            for pro_proc_gold, gold in zip(pre_proc_gold_list, gold_list):
                if gold in pre or pro_proc_gold in pre:
                    print(gold, '|||', pre)
                    return 1
                if question.find('year') > -1:
                    year_in_gold = re.search('([1-2][0-9]{3})', gold)
                    if year_in_gold is not None:
                        year_in_gold = year_in_gold.group(0)
                        if year_in_gold in pre:
                            return 1
            return 0

        sumc = 0
        total = 0
        for i in range(len(result_list)):
            key = result_list[i]['question']
            total += 1
            answer = get_answer(result_list[i]['output'])
            gold_list = temp_data_dic[key]['answers']
            pre_proc_gold_list = []
            for gold in gold_list:
                gold = unicodedata.normalize('NFKD', gold).encode('ascii', 'ignore').decode(encoding='UTF-8')
                gold = re.sub(r'\W', ' ', gold).lower().strip()
                if gold.startswith('the '):
                    gold = gold[4:]
                if gold.startswith('a '):
                    gold = gold[2:]
                if gold.startswith('an '):
                    gold = gold[3:]
                pre_proc_gold_list.append(gold)
            
            question = key.lower().strip()

            sumc += exact_match(answer, pre_proc_gold_list, gold_list, question)
        print("CWQ EM: {} / {} = {:.3f}".format(sumc, total, sumc * 1.0 / total))
        ####################