from llms.groq_llm import GroqLLM

llm = GroqLLM()
print(llm.invoke("Hello"))