# KG Construction Pipeline

1) clause_similarity_SBERT.py
- This script generates clause nodes from the 2 regulatory doc texts used in this experiment and links similar nodes based on simialrity scores

2) concept_extraction.py
- This script utilizes a pre built list of common audit terms and the clauses as a data source to extract unique healthcare auditing/regulatory concepts and link them to the nodes that MENTIONS those concepts. 
- Benefit: Concept based retireval of the graph for LLM context. If 2 clauses are linked with a similar_to and also a same concept, then it's a very strong correlation

3) finding_extraction.py
- This script ingests the on-field audit data findings and links to clause_instances to the clause nodes and department nodes of which the audit was done. Internal Audit QM.
- Benefit: to give orgniazation context and real-world scenrios to your LLM as context

# Running the Chatbot (ASK-AUDIT)

0) 
- Install dependencies as per the imports in the scripts
- create a .env with vairables: 
    NEO4J_URI = "bolt://localhost:port"
    NEO4J_USER = "your_username"
    NEO4J_PASSWORD = "your_password"
    GOOGLE_API_KEY=key
    DEEPSEEK_API_KEY=key
    GOOGLE_API_KEY_FREE=key
- You may choose any LLM of your choice
1) Run server.py in the terminal
2) Open another terminal and create a python HTTP server on a port
3) Open the chatbot.html in the browser on the server port used on 2.

# Chatbot6.py
- LangGraph used to equip the chatbot with usecase specific tools
- Usecase 1: Multi-Framework comparitive analysis. 
    Qs: Compare SHCC & JCI for hand hygeine requirements
- Usecase 2: Gap Analysis
    Qs: Identify gaps in my Adult Emergency Department Checklist
- Usecase 3: Audit Trail
    What's the latest audit findings on hand hygeine?

