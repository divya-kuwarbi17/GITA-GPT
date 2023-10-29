from sentence_transformers import SentenceTransformer, util
from vectorservices import *
from model_function import *

model2 = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
model2.max_seq_length = 512


def get_ans(query):
    query_vector = model2.encode(query)
    context = get_results(query_vector)
    prompt = create_prompt(context,query)
    answer = generate_text(prompt)
    return answer