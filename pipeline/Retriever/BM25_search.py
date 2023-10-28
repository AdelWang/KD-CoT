from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm
import os
import argparse
import multiprocessing.pool
from functools import partial


def print_results(hits):
    for i in range(len(hits)):
        print(f'{i + 1:2} {hits[i].docid:4} {hits[i].score:.5f} {hits[i].raw}')


class Bm25Searcher:
    def __init__(self, index_dir, args):
        self.index_dir = index_dir
        self.args = args
        try:
            self.searcher = LuceneSearcher(index_dir)
        except:
            print("index dir not found")
            self.searcher = LuceneSearcher.from_prebuilt_index(index_dir)
        self.searcher.set_bm25(args.k1, args.b)
        if len(args.ignore_string) > 0:
            self.ignore_list = args.ignore_string.split(',')
            print(f'ignore list: {self.ignore_list}')
        else:
            self.ignore_list = []

    def perform_search(self, data_i, top_k):

        query = data_i["query"]
        assert type(query) == type("1"), query
        for string in self.ignore_list:
            query = query.replace(string, ' ')
        query = query.strip()
        results = self.searcher.search(query, k=top_k, strip_segment_id=False, remove_dups=False)
        if len(results) < top_k:
            query = 'Find information about ' + query
            results = self.searcher.search(query, k=top_k, strip_segment_id=False, remove_dups=False)

        ctxs = []
        for result in results:
            doc_dict = json.loads(result.raw)
            ctx_text = doc_dict["contents"]
            ctx = {"title": doc_dict["title"], "text": ctx_text, "score": result.score}
            ctxs.append(ctx)

        output_i = data_i.copy()
        output_i["ctxs"] = ctxs
        return output_i


def search_all(process_idx, num_process, searcher, args):
    with open(args.query_data_path, 'r') as rf:
        data = json.load(rf)

    output_data = []
    for i, data_i in tqdm(enumerate(data)):
        if i % num_process != process_idx:
            continue
        if i > args.num_queries and args.num_queries != -1:
            break

        output_i = searcher.perform_search(data_i, args.top_k)
        output_data.append(output_i)
    return output_data

parser = argparse.ArgumentParser(description='Search using pySerini')
parser.add_argument("--index_dir", type=str,
                    help="directory to store the search index")
parser.add_argument("--query_data_path", type=str,
                    default='...',
                    help="directory to store the queries")
parser.add_argument("--output_path", type=str,
                    default='...',
                    help="directory to store the retrieved output")
parser.add_argument("--num_process", type=int, default=10,
                    help="number of processes to use for multi-threading")
parser.add_argument("--top_k", type=int, default=150,
                    help="number of passages to be retrieved for each query")
parser.add_argument("--ignore_string", type=str, default="",
                    help="string to ignore in the query, split by comma")
parser.add_argument("--b", type=float, default=0.4,
                    help="parameter of BM25")
parser.add_argument("--k1", type=float, default=0.9,
                    help="parameter of BM25")
parser.add_argument("--num_queries", type=int, default=1000,
                    help="number of queries to test, set -1 to test all")
parser.add_argument("--save", action="store_true",
                    help="whether to save the output")
args = parser.parse_args()

if __name__ == '__main__':
    index_dir = args.index_dir
    print("index dir: ", index_dir)
    searcher = Bm25Searcher(index_dir, args)

    num_process = args.num_process
    pool = multiprocessing.pool.ThreadPool(processes=num_process)
    sampleData = [x for x in range(num_process)]
    search_all_part = partial(search_all,
                              searcher=searcher,
                              num_process=num_process,
                              args=args)
    results = pool.map(search_all_part, sampleData)
    pool.close()

    output_data = []
    for result in results:
        output_data += result

    # sort the output data by question id
    output_data = sorted(output_data, key=lambda item: item['id'])
    tokenizer = None

    # save output data
    # create output dir recursively if not exist
    if args.save:
        output_path = args.output_path
        print("saving output data to {}".format(output_path))
        with open(output_path, "w") as wf:
            json.dump(output_data, wf, indent=2)

######################
# Input format(json file)
# max token number:1024
######################
# Input
# [
#     {
#         'id': id,
#         'query': query
#     }
# ]
#
# Ouput
# [
#     {
#         'id': id,
#         'query': query,
#         'ctxs': [
#             {
#                 'title': title,
#                 'text': text,
#                 'score': score
#             }
#         ]
#     }
# ]
######################