import json
from tqdm import tqdm
from conv_fin_agent import create_graph

DATA_FILE = "data/train.json"


def main():
    """
    This method uses the agent to answer the questions in the data file & assess the results.
    """
    agent = create_graph()
    print("Agent created")
    with open(DATA_FILE, "r") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents")


    # There are two types of questions in the data set. Type 2 questions have multiple questions in the same document.
    type_2_documents = [doc for doc in documents if 'qa_1' in doc]
    type_1_documents = [doc for doc in documents if 'qa_1' not in doc]

    print("Num Type 1 questions:", len(type_1_documents))
    print("Num Type 2 questions:", len(type_2_documents))

    # Answer the type 1 questions, these have a single question in the document
    type_1_questions = []
    type_1_ground_truth= []
    type_1_generated_answers = []
    for document in tqdm(type_1_documents[0:2]):
        # Context to load into the agent
        pre_text = document['pre_text']
        table = document['table']
        post_text = document['post_text']

        question_to_ask = document['qa']['question']
        answer = document['qa']['answer']


        print("QUESTION:", question_to_ask)
        print("EXPECTED ANSWER:", answer)

        result = agent.invoke(
            {"messages": [("user", f"Answer the following question; {question_to_ask}. Given the context: {pre_text}, with the table {table} and {post_text}")]}, subgraphs=True
        )

        print("RESULT:", result)

        type_1_questions.append(question_to_ask)
        type_1_ground_truth.append(answer)
        type_1_generated_answers.append(result)

        #TODO: Check if the results are as you expect
        # Not we use the second to last message in the stream as the final message is the end statement
        #assert s["messages"][-2]["content"] == answer, f"Expected {answer} but got {s['messages'][-1]['content']}"

if __name__ == "__main__":
    main()
