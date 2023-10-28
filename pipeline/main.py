from GPT_inference_steps import GPT_Infenrence
from Retriever.Retriever import DPR_retriever, BM25_retriever
from Reader.FiD_reader import FiD_reader

TEMP_FILE_DIR = './temp_file'
CWQ_PATH = 'your_cwq_test_data'
WEBQSP_PATH = 'your_webqsp_test_data'

def GPT():
    gpt_inference = GPT_Infenrence(
        temp_dir_path=TEMP_FILE_DIR,
        webqsp_path=WEBQSP_PATH,
        cwq_path=CWQ_PATH,
        temp_file_name='temp_file_demon'
    )

    ########################
    # webqsp inference steps
    ########################
    # gpt_inference.construct_webqsp_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_webqsp_response(version_num=3)
    # gpt_inference.update_iteration()

    # gpt_inference.construct_webqsp_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_webqsp_response(version_num=3)
    # gpt_inference.webqsp_eval()
    # gpt_inference.update_iteration()

    # gpt_inference.construct_webqsp_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_webqsp_response(version_num=3)
    # gpt_inference.webqsp_eval()
    # gpt_inference.update_iteration()

    # gpt_inference.construct_webqsp_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_webqsp_response(version_num=3)
    # gpt_inference.webqsp_eval()

    #####################
    # cwq inference steps
    #####################
    # gpt_inference.construct_cwq_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_cwq_response(version_num=3)
    # gpt_inference.update_iteration()

    # gpt_inference.construct_cwq_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_cwq_response(version_num=3)
    # gpt_inference.cwq_eval()
    # gpt_inference.update_iteration()

    # gpt_inference.construct_cwq_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_cwq_response(version_num=3)
    # gpt_inference.cwq_eval()
    # gpt_inference.update_iteration()

    # gpt_inference.construct_cwq_prompts(format='gpt3.5', special_demon=True, version_num=3)
    # gpt_inference.process_cwq_response(version_num=3)
    # gpt_inference.cwq_eval()
    return gpt_inference.iteration

def DPR(iteration):
    retriever = DPR_retriever(
        temp_file_dir=TEMP_FILE_DIR,
        iteration=iteration,
        temp_file_name='temp_file_demon'
    )
    #####################
    # webqsp inference steps
    #####################
    # retriever.process_webqsp_to_dpr_input()
    # retriever.split_webqsp_cwq_dpr_output()
    # retriever.prepare_for_webqsp_fid_input()
    # retriever.eval_top_k(
        # fid_input_file=TEMP_FILE_DIR + '/retriever/DPR/webqsp_fid_input' + str(iteration) + '.json'
    # )


    #####################
    # cwq inference steps
    #####################
    # retriever.process_cwq_to_dpr_input()
    # retriever.merge_webqsp_and_cwq_for_dpr_input()
    # retriever.prepare_for_fid_input()
    # retriever.eval_top_k(
    #     fid_input_file=TEMP_FILE_DIR + '/retriever/DPR/cwq_fid_input' + str(iteration) + '.json'
    # )

def BM25(iteration):
    retriever = BM25_retriever(
        temp_file_dir=TEMP_FILE_DIR,
        iteration=iteration,
        temp_file_name='temp_file_demon'
    )
    #####################
    # webqsp inference steps
    #####################
    # retriever.process_webqsp_to_BM25_input()
    # retriever.prepare_for_webqsp_fid_input()
    # retriever.eval_top_k(
    #     fid_input_file=TEMP_FILE_DIR + '/retriever/BM25/webqsp_fid_input' + str(iteration) + '.json'
    # )


    #####################
    # cwq inference steps
    #####################
    # retriever.process_cwq_to_BM25_input()
    # retriever.merge_and_prepare_for_fid_input(['cwq_for_bm25_output0_freebase.json', 'cwq_for_bm25_output0_wiki.json'])
    # retriever.prepare_for_fid_input()
    # retriever.eval_top_k(
    #     fid_input_file=TEMP_FILE_DIR + '/retriever/BM25/cwq_fid_input' + str(iteration) + '.json'
    # )

def FiD(iteration):
    reader = FiD_reader(
        temp_file_dir=TEMP_FILE_DIR,
        iteration=iteration,
        temp_file_name='temp_file_demon'
    )
    ########################
    # webqsp inference steps
    ########################
    # reader.get_webqsp_intermediate_answer(
    #     fid_output_path=TEMP_FILE_DIR + '/reader/fid_output/webqsp_' + str(iteration + 1) + '/final_output.txt'
    # )

    #####################
    # cwq inference steps
    #####################
    # reader.get_cwq_intermediate_answer(
    #     fid_output_path=TEMP_FILE_DIR + '/reader/fid_output/cwq_' + str(iteration + 1) + '/final_output.txt'
    # )

if __name__ == '__main__':
    iter = GPT()
    DPR(iter)
    # BM25(iter)
    FiD(iter)