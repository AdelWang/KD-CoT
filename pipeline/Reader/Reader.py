import os
import json
import pandas as pd

class FiD_reader():
    def __init__(self, temp_file_dir : str, iteration : int, temp_file_name: str = 'temp_file'):
        self.temp_file_dir = temp_file_dir
        self.webqsp_temp_file = os.path.join(self.temp_file_dir, 'webqsp_' + temp_file_name + '.json')
        self.cwq_temp_file = os.path.join(self.temp_file_dir, 'cwq_' + temp_file_name + '.json')
        self.iteration = iteration

    ###################
    # webqsp processing
    ###################
    def get_webqsp_intermediate_answer(self, fid_output_path : str):
        temp_data = json.load(open(self.webqsp_temp_file))

        with open(fid_output_path) as f:
            lines = f.readlines()
            lines = [line.split('\t') for line in lines]
            fid_results = [{'id': line[0].strip().rstrip(), 'answer': line[1].strip().rstrip()} for line in lines]

        temp_data_by_id = {q['id']: q for q in temp_data}
        print("fid result number:", len(fid_results))
        for i in range(len(fid_results)):
            id = fid_results[i]['id']
            if pd.isnull(fid_results[i]['answer']):
                fid_results[i]['answer'] = 'Unkown'
            fid_results[i]['answer'] = str(fid_results[i]['answer'])
            temp_data_by_id[id]['IntermediateFiDAnswer'].append(','.join([fid_results[i]['answer']]))
            assert len(temp_data_by_id[id]['IntermediateFiDAnswer']) == self.iteration + 1, "Answer number doesn't match iteration!" + str(temp_data_by_id[id]['IntermediateAnswer'])

        temp_data = [value for key, value in temp_data_by_id.items()]
        json.dump(temp_data, open(self.webqsp_temp_file, 'w'), indent=2)

    ################
    # cwq processing
    ################
    def get_cwq_intermediate_answer(self, fid_output_path: str):
        temp_data = json.load(open(self.cwq_temp_file))

        with open(fid_output_path) as f:
            lines = f.readlines()
            lines = [line.split('\t') for line in lines]
            fid_results = [{'id': line[0].strip().rstrip(), 'answer': line[1].strip().rstrip()} for line in lines]

        temp_data_by_id = {q['id']: q for q in temp_data}
        print("fid result number:", len(fid_results))
        for i in range(len(fid_results)):
            id = fid_results[i]['id']
            if pd.isnull(fid_results[i]['answer']):
                fid_results[i]['answer'] = 'Unkown'
            fid_results[i]['answer'] = str(fid_results[i]['answer'])
            temp_data_by_id[id]['IntermediateFiDAnswer'].append(','.join([fid_results[i]['answer']]))
            assert len(temp_data_by_id[id]['IntermediateFiDAnswer']) == self.iteration + 1, "Answer number doesn't match iteration!" + str(temp_data_by_id[id]['IntermediateAnswer'])

        temp_data = [value for key, value in temp_data_by_id.items()]
        json.dump(temp_data, open(self.cwq_temp_file, 'w'), indent=2)