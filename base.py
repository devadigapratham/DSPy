import dspy 

# Use the exact model name "llama3.2:3b"
lm = dspy.LM("ollama_chat/llama3.2:3b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)

# # Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
# qa = dspy.ChainOfThought("question -> answer")
# response = qa(question="How many floors are available in the Burj Khalifa?")
# print(response.answer)


qa = dspy.ChainOfThought("question -> answer")
dspy.configure(lm=dspy.LM('ollama_chat/llama3.2:3b'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('Llama 3.2 :', response.answer)

with dspy.context(lm=dspy.LM('ollama_chat/mistral:latest')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('Mistral:', response.answer)