import os
import json
import csv
from collections import defaultdict
from tqdm.auto import tqdm
import unicodedata
import regex as re

def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        if tokenizer is None:
            text = text.lower()
            for single_answer in answers:
                if not single_answer:
                    single_answer = '-1'
                norm_answer = _normalize(single_answer).lower()
                if norm_answer in text:
                    return True
        else:
            text = tokenizer.tokenize(text).words(uncased=True)

            for single_answer in answers:
                single_answer = _normalize(single_answer)
                single_answer = tokenizer.tokenize(single_answer)
                single_answer = single_answer.words(uncased=True)

                for i in range(0, len(text) - len(single_answer) + 1):
                    if single_answer == text[i : i + len(single_answer)]:
                        return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False

def _normalize(text):
    return unicodedata.normalize("NFD", text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None

class BM25_retriever():
    def __init__(self, temp_file_dir : str, iteration : int, temp_file_name: str = 'temp_file'):
        self.temp_file_dir = temp_file_dir
        self.webqsp_temp_file = os.path.join(self.temp_file_dir, 'webqsp_' + temp_file_name + '.json')
        self.cwq_temp_file = os.path.join(self.temp_file_dir, 'cwq_' + temp_file_name + '.json')
        self.iteration = iteration
        self.bm25_dir = os.path.join(self.temp_file_dir, 'retriever', 'BM25')
        if not os.path.exists(self.bm25_dir):
            os.makedirs(self.bm25_dir)
    
    ######
    # CWQ
    ######

    def process_cwq_to_BM25_input(self):
        temp_data = json.load(open(self.cwq_temp_file))
        
        result_list = []
        for data in temp_data:
            if data['PredictedAnswer'] != -1:
                continue
            result = {
                'id': data['id'],
                'query': data['IntermediateAction'][-1][1]
            }
            result_list.append(result)
        print('BM25 question number:', len(result_list))
        output_path = os.path.join(self.bm25_dir, 'cwq_for_bm25_input' + str(self.iteration) + '.json')
        json.dump(result_list, open(output_path, 'w'), indent=2)
    
    def prepare_for_fid_input(self): # CWQ
        temp_data = json.load(open(self.cwq_temp_file))
        temp_data_dic = {data['id']: data for data in temp_data}

        bm25_result_path = os.path.join(self.bm25_dir, 'cwq_for_bm25_output' + str(self.iteration) + '.json')
        bm25_result_list = json.load(open(bm25_result_path))

        fid_input = []
        for result in bm25_result_list:
            key = result['id']
            assert key in temp_data_dic, key
            type = 'single' if temp_data_dic[key]['IntermediateAction'][-1][0] == 'Question' else 'multi'
            fid_input.append({
                'id': key,
                'question': result['query'],
                'type': type,
                'answers': temp_data_dic[key]['answers'],
                'ctxs': result['ctxs']
            })
        
        print('CWQ Fid input data number:', len(fid_input))
        output_path = os.path.join(self.bm25_dir, 'cwq_fid_input' + str(self.iteration) + '.json')
        json.dump(fid_input, open(output_path, 'w'), indent=2)

    
    #########
    # WEBQSP
    #########
    
    def process_webqsp_to_BM25_input(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        
        result_list = []
        for data in temp_data:
            if data['PredictedAnswer'] != -1:
                continue
            result = {
                'id': data['id'],
                'query': data['IntermediateAction'][-1][1]
            }
            result_list.append(result)
        print('BM25 question number:', len(result_list))
        output_path = os.path.join(self.bm25_dir, 'webqsp_for_bm25_input' + str(self.iteration) + '.json')
        json.dump(result_list, open(output_path, 'w'), indent=2)
    
    def prepare_for_webqsp_fid_input(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        temp_data_dic = {data['id']: data for data in temp_data}

        bm25_result_path = os.path.join(self.bm25_dir, 'webqsp_for_bm25_output' + str(self.iteration) + '.json')
        bm25_result_list = json.load(open(bm25_result_path))

        fid_input = []
        for result in bm25_result_list:
            key = result['id']
            assert key in temp_data_dic, key
            type = 'single' if temp_data_dic[key]['IntermediateAction'][-1][0] == 'Question' else 'multi'
            fid_input.append({
                'id': key,
                'question': result['query'],
                'type': type,
                'answers': temp_data_dic[key]['answers'],
                'ctxs': result['ctxs']
            })
        
        print('WEBQSP Fid input data number:', len(fid_input))
        output_path = os.path.join(self.bm25_dir, 'webqsp_fid_input' + str(self.iteration) + '.json')
        json.dump(fid_input, open(output_path, 'w'), indent=2)
    
    def eval_top_k_one(self, data_i, top_k=100, tokenizer=None):
        recall = 0
        answers = data_i['answers']
        # answers = data_i['short_answers']
        for answer in answers:
            for ctx in data_i['ctxs'][:top_k]:
                context = ctx['text']
                if has_answer([answer], context, tokenizer, "string"):
                    recall += 1
                    break
        return recall / (len(answers) + 1e-8)
    
    def eval_top_k(self, fid_input_file, top_k_list=[20, 100], tokenizer=None):
        output_data = json.load(open(fid_input_file))

        print("Evaluation")
        hits_dict = defaultdict(int)
        recall_dict = defaultdict(float)
        top_k_list = [k for k in top_k_list if k <= len(output_data[0]['ctxs'])]
        for data_i in tqdm(output_data):
            for k in top_k_list:
                recall = self.eval_top_k_one(data_i, top_k=k, tokenizer=tokenizer)
                if recall > 0:
                    hits_dict[k] += 1
                recall_dict[k] += recall
        for k in top_k_list:
            print("Top {}".format(k),
                  "Hits: ", round(hits_dict[k] * 100 / len(output_data), 1),
                  "Recall: ", round(recall_dict[k] * 100 / len(output_data), 1))
            

class DPR_retriever():
    def __init__(self, temp_file_dir : str, iteration : int, temp_file_name: str = 'temp_file', special = None):
        self.temp_file_dir = temp_file_dir
        self.webqsp_temp_file = os.path.join(self.temp_file_dir, 'webqsp_' + temp_file_name + '.json')
        self.cwq_temp_file = os.path.join(self.temp_file_dir, 'cwq_' + temp_file_name + '.json')
        self.iteration = iteration
        self.dpr_dir = os.path.join(self.temp_file_dir, 'retriever', 'DPR')
        if special:
            self.dpr_dir = os.path.join(self.dpr_dir, special)
        if not os.path.exists(self.dpr_dir):
            os.makedirs(self.dpr_dir)
    
    ######
    # CWQ
    ######

    def process_cwq_to_dpr_input(self):
        temp_data = json.load(open(self.cwq_temp_file))
        
        result_list = []
        for data in temp_data:
            if data['PredictedAnswer'] != -1:
                continue
            result = {
                'id': data['id'],
                'question': data['IntermediateAction'][-1][1],
                'answers': data['answers']
            }
            result_list.append(result)
        print('CWQ DPR question number:', len(result_list))
        output_path = os.path.join(self.dpr_dir, 'cwq_for_dpr_input' + str(self.iteration) + '.json')
        json.dump(result_list, open(output_path, 'w'), indent=2)
    
    def prepare_for_fid_input(self): # CWQ
        temp_data = json.load(open(self.cwq_temp_file))
        temp_data_dic = {data['id']: data for data in temp_data}

        dpr_result_path = os.path.join(self.dpr_dir, 'cwq_for_dpr_output' + str(self.iteration) + '.json')
        dpr_result_list = json.load(open(dpr_result_path))

        fid_input = []
        for result in dpr_result_list:
            key = result['id']
            assert key in temp_data_dic, key
            type = 'single' if temp_data_dic[key]['IntermediateAction'][-1][0] == 'Question' else 'multi'
            fid_input.append({
                'id': key,
                'question': result['question'],
                'type': type,
                'answers': temp_data_dic[key]['answers'],
                'ctxs': result['ctxs']
            })
        
        print('CWQ Fid input data number:', len(fid_input))
        output_path = os.path.join(self.dpr_dir, 'cwq_fid_input' + str(self.iteration) + '.json')
        json.dump(fid_input, open(output_path, 'w'), indent=2)
    
    #########
    # WEBQSP
    #########
    
    def process_webqsp_to_dpr_input(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        
        result_list = []
        for data in temp_data:
            if data['PredictedAnswer'] != -1:
                continue
            result = {
                'id': data['id'],
                'question': data['IntermediateAction'][-1][1],
                'answers': data['answers']
            }
            result_list.append(result)
        print('WEBQSP DPR question number:', len(result_list))
        output_path = os.path.join(self.dpr_dir, 'webqsp_for_dpr_input' + str(self.iteration) + '.json')
        json.dump(result_list, open(output_path, 'w'), indent=2)
    
    def prepare_for_webqsp_fid_input(self):
        temp_data = json.load(open(self.webqsp_temp_file))
        temp_data_dic = {data['id']: data for data in temp_data}

        dpr_result_path = os.path.join(self.dpr_dir, 'webqsp_for_dpr_output' + str(self.iteration) + '.json')
        dpr_result_list = json.load(open(dpr_result_path))

        fid_input = []
        for result in dpr_result_list:
            key = result['id']
            assert key in temp_data_dic, key
            type = 'single' if temp_data_dic[key]['IntermediateAction'][-1][0] == 'Question' else 'multi'
            fid_input.append({
                'id': key,
                'question': result['question'],
                'type': type,
                'answers': temp_data_dic[key]['answers'],
                'ctxs': result['ctxs']
            })
        
        print('WEBQSP Fid input data number:', len(fid_input))
        output_path = os.path.join(self.dpr_dir, 'webqsp_fid_input' + str(self.iteration) + '.json')
        json.dump(fid_input, open(output_path, 'w'), indent=2)
    
    def merge_webqsp_and_cwq_for_dpr_input(self):
        cwq_path = os.path.join(self.dpr_dir, 'cwq_for_dpr_input' + str(self.iteration) + '.json')
        webqsp_path = os.path.join(self.dpr_dir, 'webqsp_for_dpr_input' + str(self.iteration) + '.json')
        cwq_in_data = json.load(open(cwq_path))
        webqsp_in_data = json.load(open(webqsp_path))
        merge_data = cwq_in_data + webqsp_in_data
        output_path = os.path.join(self.dpr_dir, 'merge_dpr_input' + str(self.iteration) + '.json')
        json.dump(merge_data, open(output_path, 'w'), indent=2)
    
    def split_webqsp_cwq_dpr_output(self):
        cwq_path = os.path.join(self.dpr_dir, 'cwq_for_dpr_input' + str(self.iteration) + '.json')
        webqsp_path = os.path.join(self.dpr_dir, 'webqsp_for_dpr_input' + str(self.iteration) + '.json')
        cwq_in_data = json.load(open(cwq_path))
        webqsp_in_data = json.load(open(webqsp_path))
        cwq_in_ids = [data['id'] for data in cwq_in_data]
        webqsp_in_ids = [data['id'] for data in webqsp_in_data]

        cwq_out_data = []
        webqsp_out_data = []
        
        merge_path = os.path.join(self.dpr_dir, 'merge_dpr_output' + str(self.iteration) + '.json')
        merge_data = json.load(open(merge_path))
        for data in merge_data:
            id = data['id']
            assert id in cwq_in_ids or id in webqsp_in_ids, 'ID error!'
            if id in cwq_in_ids:
                cwq_out_data.append(data)
            else:
                webqsp_out_data.append(data)
        
        cwq_output_path = os.path.join(self.dpr_dir, 'cwq_for_dpr_output' + str(self.iteration) + '.json')
        json.dump(cwq_out_data, open(cwq_output_path, 'w'), indent=2)
        webqsp_output_path = os.path.join(self.dpr_dir, 'webqsp_for_dpr_output' + str(self.iteration) + '.json')
        json.dump(webqsp_out_data, open(webqsp_output_path, 'w'), indent=2)

    def eval_top_k_one(self, data_i, top_k=100, tokenizer=None):
        recall = 0
        answers = data_i['answers']
        for answer in answers:
            for ctx in data_i['ctxs'][:top_k]:
                context = ctx['text']
                if has_answer([answer], context, tokenizer, "string"):
                    recall += 1
                    break
        return recall / (len(answers) + 1e-8)

    def eval_top_k(self, fid_input_file, top_k_list=[20, 100], tokenizer=None):
        output_data = json.load(open(fid_input_file))

        print("Evaluation")
        hits_dict = defaultdict(int)
        recall_dict = defaultdict(float)
        top_k_list = [k for k in top_k_list if k <= len(output_data[0]['ctxs'])]
        for data_i in tqdm(output_data):
            for k in top_k_list:
                recall = self.eval_top_k_one(data_i, top_k=k, tokenizer=tokenizer)
                if recall > 0:
                    hits_dict[k] += 1
                recall_dict[k] += recall
        for k in top_k_list:
            print("Top {}".format(k),
                  "Hits: ", round(hits_dict[k] * 100 / len(output_data), 1),
                  "Recall: ", round(recall_dict[k] * 100 / len(output_data), 1))