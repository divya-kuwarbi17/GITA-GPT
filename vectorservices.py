from pymilvus import connections,utility,Collection,CollectionSchema, FieldSchema,DataType

connections.connect(host="localhost",port="19530",db_name = 'Gitagpt')
#,using = "default")
print(utility.list_collections())
collection = Collection(name = "divyaGita")
def get_results(query_vector):
    results = collection.search(data = [query_vector], anns_field="embedding",limit = 10,param ={"metric_type":"IP"},output_fields=['passages'] )
    a = ""
    for i in results[0]:
        a = a + i.entity.get('passages')
    return a

def create_prompt(context,question):
    prompt = f"""Answer the question as truthfully and specific as possible using the provided context. Be precise as much as possible.
    Note : It is Important that answer should be from the given context if context is not enough to answer the question then reply i don't know
    Context : {context}.
    Question : {question}?"""
    return prompt