"""
This is just a short pilot test to see if the model is able to answer questions.
"""

import sys
import json
import openai


# Load the secret key
with open("SECRET_KEY.json", encoding="utf-8") as f:
    data = json.load(f)
openai.api_key = data["API_KEY"]


def gpt_3_chat(query: str) -> str:
    """
    Connect to OpenAI and ask a question.
    :query: string, the question to ask
    :return: string, the answer
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query,
        temperature=0.8,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    # Just to debug:
    # print(type(response))
    # print(response)

    if "choices" in response:
        if len(response["choices"]) > 0:
            answer = response["choices"][0]["text"]
        else:
            # If there are no choices, then the model is not able
            # to answer the question
            answer = "Opps sorry, you beat the AI this time"
    else:
        # If there are no choices, then the model is not able to
        # answer the question
        answer = "Opps sorry, you beat the AI this time"
    #print(answer)

    return answer

# for testing the code in VC (remember to uncomment the print statement)
# gpt_3_chat("Is there a good solution to connect GPT-3 to miro.com?")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        parameters = sys.argv[1]
        parameters += " . Please, answer something sarcastic or/and funny to this question."
        print(gpt_3_chat(parameters))
    else:
        print("ERROR: Please provide a query! The query should be \
            in the form of a question.")
