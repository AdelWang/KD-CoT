import jsonlines
import json
import re
import os

original_prompt = '''Which is the correct answer for the question {question}\n{choice_1} or {choice_2}\nAssistant:'''

match_prompt = '''For the question {question}, is the answer {choice} correct? Response: '''

def cleantxt(raw):
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.,"]+', re.UNICODE)
    raw = fil.sub('', raw)
    return raw.lower()

def get_justify_model_input(temp_file_path, mode='cwq', iter=0):
    temp_data = json.load(open(temp_file_path))

    prompt_list = []
    for data in temp_data:
        prompt = original_prompt

        if data['PredictedAnswer'] != -1:
            continue
        id = data['id']
        question = data['IntermediateAction'][-1][1]
        answer1 = data['IntermediateGPTAnswer'][-1]
        answer2 = data['IntermediateFiDAnswer'][-1]
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{choice_1}', answer1)
        prompt = prompt.replace('{choice_2}', answer2)
        prompt_list.append({
            'id': id,
            'input': prompt,
            'target': 'unk'
        })
    
    print('Justification number:', len(prompt_list))
    outdir = 'your_temp_file_dir/' + mode + '_justifier'
    outpath = os.path.join(outdir, mode + '_jus' + str(iter) + '.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)
    
    outpath = os.path.join(outdir, 'train.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)
    outpath = os.path.join(outdir, 'dev.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)


def process_justify_results(temp_file_path, results_path):
    temp_data = json.load(open(temp_file_path))
    results = json.load(open(results_path))
    results_dic = {cleantxt(result['input']): result for result in results}

    def get_answer(text, gpt_answer, fid_answer):
        if 'Neither' in text:
            if text in 'Neither, the correct answer should be ':
                return gpt_answer
            text = text.replace('Neither, the correct answer should be', '')
            return text.strip().rstrip()
        if text in gpt_answer and text in fid_answer:
            return fid_answer if len(fid_answer) < len(gpt_answer) else gpt_answer
        elif text in fid_answer:
            return fid_answer
        else:
            return gpt_answer

    total = 0
    for i, data in enumerate(temp_data):
        prompt = original_prompt

        if data['PredictedAnswer'] != -1:
            continue
        total += 1

        question = data['IntermediateAction'][-1][1]
        answer1 = data['IntermediateGPTAnswer'][-1]
        answer2 = data['IntermediateFiDAnswer'][-1]
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{choice_1}', answer1)
        prompt = prompt.replace('{choice_2}', answer2)
        o = prompt
        prompt = cleantxt(prompt)

        assert prompt in results_dic, o
        answer = get_answer(results_dic[prompt]['output'], answer1, answer2)
        temp_data[i]['IntermediateAnswer'].append(answer)
    
    print('Processed data:', total)
    json.dump(temp_data, open(temp_file_path, 'w'), indent=2)

def add_fid_answer(temp_file_path):
    temp_data = json.load(open(temp_file_path))
    total = 0
    for i, data in enumerate(temp_data):
        if data['PredictedAnswer'] != -1:
            continue
        total += 1
        fid_answer = data['IntermediateFiDAnswer'][-1]
        temp_data[i]['IntermediateAnswer'].append(fid_answer)
    
    print('Processed data:', total)
    json.dump(temp_data, open(temp_file_path, 'w'), indent=2)

def get_match_input(temp_file_path, mode='cwq', iter=0):
    temp_data = json.load(open(temp_file_path))

    prompt_list = []
    for data in temp_data:
        prompt = match_prompt

        if data['PredictedAnswer'] != -1:
            continue
        id = data['id']
        question = data['IntermediateAction'][-1][1]
        gpt_answer = data['IntermediateGPTAnswer'][-1]
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{choice}', gpt_answer)
        prompt_list.append({
            'id': id,
            'input': prompt,
            'target': 'unk'
        })
    
    print('Match number:', len(prompt_list))
    outdir = 'your_temp_file_dir/' + mode + '_match'
    outpath = os.path.join(outdir, mode + '_match' + str(iter) + '.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)
    
    outpath = os.path.join(outdir, 'train.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)
    outpath = os.path.join(outdir, 'dev.json')
    with jsonlines.open(outpath, 'w') as writer:
        for prompt in prompt_list:
            writer.write(prompt)

def process_match_results(temp_file_path, results_path):
    temp_data = json.load(open(temp_file_path))
    results = json.load(open(results_path))
    results_dic = {cleantxt(result['input']): result for result in results}

    def get_answer(text, gpt_answer):
        if 'No' in text:
            if text in 'No, the answer should be ':
                return gpt_answer
            text = text.replace('No, the answer should be', '')
            return text.strip().rstrip()
        else:
            return gpt_answer

    total = 0
    for i, data in enumerate(temp_data):
        prompt = match_prompt

        if data['PredictedAnswer'] != -1:
            continue
        total += 1

        question = data['IntermediateAction'][-1][1]
        gpt_answer = data['IntermediateGPTAnswer'][-1]
        prompt = prompt.replace('{question}', question)
        prompt = prompt.replace('{choice}', gpt_answer)
        prompt = cleantxt(prompt)

        assert prompt in results_dic, prompt
        answer = get_answer(results_dic[prompt]['output'], gpt_answer)
        temp_data[i]['IntermediateAnswer'].append(answer)
    
    print('Processed data:', total)
    json.dump(temp_data, open(temp_file_path, 'w'), indent=2)

if __name__ == '__main__':
    # !!!!
    iteration = 0
    for mode in ['webqsp', 'cwq']:
        temp_file_path = 'your_temp_file_dir/' + mode + '_temp_file_demon.json'
        # get_justify_model_input(temp_file_path=temp_file_path, mode = mode, iter = iteration)
        # process_justify_results(
        #     temp_file_path=temp_file_path, 
        #     results_path='your_temp_file_dir/' + mode + '_justifier_llama/results' + str(iteration) + '/preds_for_eval.json'
        # )


        # add_fid_answer(
        #     temp_file_path=temp_file_path
        # )

        # get_match_input(temp_file_path=temp_file_path, mode = mode, iter = iteration)
        # process_match_results(
        #     temp_file_path=temp_file_path, 
        #     results_path='your_temp_file_dir/' + mode + '_match/results' + str(iteration) + '/preds_for_eval.json'
        # )