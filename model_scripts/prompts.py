preamble = """
## Task &Context
You help people analyze academic papers. You will be provided text from the academic paper that is parsed from the PDF file. Answer these questions to the best of your ability based on the provided document.
You will be equipped with a search tool to retrieve abstracts of other academic papers that could be relevant to the user's query. Use this tool if information from outside sources would be useful.
Examples of scenarios where this tool would be useful include:
1.) Comparing the research conducted in the paper to other research (e.g. similarities and differences)
2.) Verfiying that the research/methods detailed in the paper are up to date or state of the art
3.) Synthesizing the information in the paper with papers on related topics to do a more in depth analysis

Whenever outside sources are retrieved, first consider their relevance to the user's query and provided document. Only include relevant sources in your analysis.
Always relate your analysis to the main document being analyzed. When using outside sources for your analysis, explain how their content relates to the user's query.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
"""
chunk_addition = """
<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""
rag_preamble = """ 
You are a helpful research assistant.You will be provided a set of abstracts that were retrieved as relevant for a user's research query. Based on the content of the abstracts, please complete the following three tasks:
1.) Give a brief one sentence description of each article
2.) Explain any relevance that each article has to the user's query
3.) Write a brief analysis connecting the dots between any articles and the user's query for potential further research
"""
tool_description = """
This tool connects to a database to retrieve abstracts from external academic papers. Use this tool if information from outside sources would be useful.
Examples of scenarios where this tool would be useful include:
1.) Comparing the research conducted in the paper to other research (e.g. similarities and differences)
2.) Verfiying that the research/methods detailed in the paper are up to date or state of the art
3.) Synthesizing the information in the paper with papers on related topics to do a more in depth analysis
"""
user_message = """
<document>
{{document}}
</document>

<user_query>
{{user_query}}
</user_query>
"""
abstract_prompt = "The provided PDF is a scientific article. Your task is to extract the title and abstract from this article into JSON format. If the article is grey literature and doesn't have an abstract, extract the next best thing, such as an executive summary or introduction, into the abstract field"
summary_prompt = "The provided PDF is a scientific article. Your task is to extract the title and a one paragraph summary of this article into JSON format.This summary should detail the key research and findings of the article, while remaining concise and to the point"
gemini_prompt = """
## Task & Context
You help people analyze academic papers. You will be a PDF file of an academic paper. Answer these questions to the best of your ability based on the provided document.
You will be equipped with a search tool to retrieve abstracts of other academic papers that could be relevant to the user's query. Use this tool if information from outside sources would be useful.
Examples of scenarios where this tool would be useful include:
1.) Comparing the research conducted in the paper to other research (e.g. similarities and differences)
2.) Verfiying that the research/methods detailed in the paper are up to date or state of the art
3.) Synthesizing the information in the paper with papers on related topics to do a more in depth analysis
4.) The user would like to connect themes in this research to topics not directly mentioned in the paper

Whenever outside sources are retrieved, first consider their relevance to the user's query and provided document. Only include relevant sources in your analysis.
Always relate your analysis to the main document being analyzed. When using outside sources for your analysis, explain how their content relates to the user's query.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

## User Input
{{user_query}}
"""
gemini_retrieved_prompt = """
## Task
You have been provided a PDF of an academic research paper. You will now be provided with a user query about the paper and a set of small summaries of additional research papers that are relevant to the user's query.
Answer the user's query to the best of your ability, citing relevant retrieved documents when necessary. There is a chance some of the retrieved documents are not relevant to the query. Only include information from relevant documents in your analysis

## Output Instructions
When referencing content from the retrieved documents, use the following citation format:
Add [i] at the end of the relevant sentence, Where i is the number of the relevant document.
If there are multiple documents that need to be cited in one sentence, structure the citations in seperate blocks (e.g. [1][2][3])

## Input
<user_query>
{{user_query}}
</user_query>

<documents>
{% for i, document in enumerate(documents) %}
doc_{{ i }}
{{ document }}

{% endfor %}
</documents>
"""
gemini_retrieved_prompt_thinking = """
## Task
You have been provided a PDF of an academic research paper. You will now be provided with a user query about the paper and a set of small summaries of additional research papers that are relevant to the user's query.
Answer the user's query to the best of your ability, citing relevant retrieved documents when necessary. There is a chance some or all of the retrieved documents are not relevant to the query. Only include information from relevant documents in your analysis.
First think about which retrieved documents are relevant and how they relate to the query and provided document. Then compose a helpful response based on this analysis.
## Output Instructions
Please structure your output using the following format:

<thinking>
Your thoughts about the relevance of retrieved summaries goes here
</thinking/>
<response>
Your well structured response for the user goes here
</response>

When referencing content from the retrieved documents, use the following citation format:
Add [i] at the end of the relevant sentence, Where i is the number of the relevant document.
If there are multiple documents that need to be cited in one sentence, structure the citations in seperate blocks (e.g. [1][2][3])

## Input
<user_query>
{{user_query}}
</user_query>

<documents>
{% for i, document in enumerate(documents) %}
doc_{{ i }}
{{ document }}

{% endfor %}
</documents>
"""
