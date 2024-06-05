

from langchain_community.llms import Ollama
from langchain_community.retrievers import WikipediaRetriever

import pandas as pd
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from sklearn.metrics import classification_report


def senloader(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def save_results(path: str, headline:str, entity:str, true_label:str, output: str) -> None: #TODO
    with open(path, "a") as resultsfile:
        resultsfile.write("\n---------\n")
        resultsfile.write("HEADLINE\n")
        resultsfile.write(headline)
        resultsfile.write("\nENTITY\n")
        resultsfile.write(entity)
        resultsfile.write("\nTRUE LABEL\n")
        resultsfile.write(true_label)
        resultsfile.write("\nPREDICTION\n")
        resultsfile.write(output[0])
        resultsfile.write("\nLLM OUTPUT\n")
        resultsfile.write(headline)


def parse(message: str) -> [str, str]:
    if "Label: Negative" in message: return ["Negative", message]
    if "Label: Positive" in message: return ["Positive", message]
    if "Label: Neutral" in message: return ["Neutral", message]
    if "negative" in message: return ["Negative", message]
    if "NEGATIVE" in message: return ["Negative", message]
    if "positive" in message: return ["Positive", message]
    if "POSITIVE" in message:
        return ["Positive", message]
    else:
        return ["Neutral", message]


def format_docs(docs):
    return docs[0].metadata["summary"]


def examples():
    return [{"headline": "Trump is a great president",
             "entity": "Trump",
             "context": "Trump is a former president of United States of America.",
             "label": "Positive"},

            {"headline": "Biden is a great president",
             "entity": "Biden",
             "context": "Biden is a president of United States of America",
             "label": "Positive"}
            ]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    llm = Ollama(model="llama2")

    # chain = create_tagging_chain(schema=schema, llm=llm)

    data = senloader("/home/kasia/experiments/sen/SEN/senen.csv")

    retriever = WikipediaRetriever(search_kwargs={"k": 1})

    entities_contex = {}

    for ent in data["entity"].unique():
        if ent not in entities_contex.keys():
            entities_contex[ent] = retriever.get_relevant_documents(query=ent)[0].metadata["summary"]

    example_prompt = PromptTemplate(
        input_variables=["headline", "entity", "context", "label"],
        template="""Predict sentiment expressed in the given news headline towards given entity.   
        You have to be objective towards all entities. 
        
            Headline: {headline}
            Entity: {entity}
            Context: {context} 

            Give your answer as exactly one label: Positive, Negative, Neutral.
            
            Label: {label}"""
    )

    print(example_prompt.format(**examples()[0]))

    prompt = FewShotPromptTemplate(
        examples=examples(),
        example_prompt=example_prompt,
        suffix="""Predict sentiment expressed in the given news headline towards given entity. 
        You have to be objective towards all entities. 
        
            Headline: {headline}
            Entity: {entity}
            Context: {context} 
            
            Give your answer as exactly one label: Positive, Negative, Neutral. 
            
            Label: """,
        input_variables=["headline", "entity", "context"],
    )

    # print(prompt.format(input="Who was the father of Mary Ball Washington?"))
    predictions = []
    true_labels = []

    for i in range(len(data)):
        senti = data["Sentiment"][i]
        if senti != "Unknown":
            headline = data["headline"][i]
            entity = data["entity"][i]
            true_labels.append(senti)

            rag_chain = (
                    prompt
                    | llm
                    | parse
            )
            output = rag_chain.invoke({"headline": headline, "entity": entity,
                                       "context": entities_contex[entity]})

            predictions.append(output[0])
            save_results("/home/kasia/PycharmProjects/langchain/results/output.txt",headline,entity,senti,output)

    print("True")
    print(true_labels)
    print(predictions)
    print(classification_report(true_labels, predictions))
    with open("/home/kasia/PycharmProjects/langchain/results/results.txt", "w") as resultsfile:
        resultsfile.writelines(true_labels)
        resultsfile.writelines(predictions)
        resultsfile.writelines(classification_report(true_labels, predictions))