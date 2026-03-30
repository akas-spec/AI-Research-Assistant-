"""
llms/prompts.py - Centralized prompt templates
"""

# Query classification prompt
CLASSIFICATION_PROMPT = """
You are a query classifier. Analyze the user's question and classify it into ONE category:

1. SIMPLE: General questions not requiring document retrieval
   Examples: "What is a transformer?", "Explain attention mechanism", "Define RAG"
   
2. RETRIEVAL: Questions needing specific information from papers
   Examples: "What does paper X say about Y?", "Find papers on topic Z", "What are recent findings on X?"
   
3. ANALYTICAL: Questions comparing/analyzing multiple papers or concepts
   Examples: "Compare approaches in these papers", "What are differences between X and Y?", "Analyze the evolution of X"
   
4. RESEARCH: Complex questions needing comprehensive analysis and synthesis
   Examples: "What's the state of research on X?", "Synthesize findings across the field", "What are research gaps in X?"

User Query: {query}

Respond with ONLY the category name (SIMPLE, RETRIEVAL, ANALYTICAL, or RESEARCH) followed by a pipe and brief reason.
Format: CATEGORY | reason

Example: RETRIEVAL | User is asking for specific information from papers
"""

# Simple query prompt
SIMPLE_PROMPT = """
You are a helpful AI research assistant. Answer the following question clearly and concisely.

Question: {query}

Provide a clear, accurate answer. If the question is about research concepts, explain them in an accessible way.
"""

# Retrieval query prompt
RETRIEVAL_PROMPT = """
Context from research papers:
{context}

User Question: {query}

Based on the context provided above, answer the user's question. 
- Be specific and cite information from the context
- If the context doesn't fully answer the question, acknowledge this
- Use clear, structured formatting

Answer:
"""

# Analytical query prompt
ANALYTICAL_PROMPT = """
You are analyzing multiple research sources to answer a complex question.

Retrieved Information:
{context}

User Question: {query}

Provide a comprehensive analytical answer that:
1. Compares and contrasts different approaches/findings
2. Identifies key similarities and differences
3. Highlights important insights
4. Draws connections between sources
5. Provides critical analysis

Use clear structure with headings if helpful. Cite sources when making specific claims.

Analysis:
"""

# Research synthesis prompt
RESEARCH_SYNTHESIS_PROMPT = """
You are synthesizing research findings to provide a comprehensive overview.

Original Research Question: {query}

Extracted Information from Papers:
{extracted_info}

Provide a thorough research synthesis that includes:

1. **Current State of Research**: Summarize what is currently known
2. **Key Approaches & Methodologies**: Describe main research approaches
3. **Important Findings**: Highlight significant results and contributions
4. **Research Gaps & Future Directions**: Identify what's missing and where the field is heading
5. **Critical Insights**: Provide thoughtful analysis and takeaways

Write in a clear, academic style with proper structure. Use evidence from the papers.

Synthesis:
"""

# Extraction prompt (for research pipeline)
EXTRACTION_PROMPT = """
Research papers content:
{papers_content}

Extract the following information from these papers:
1. Main contributions and innovations
2. Methodologies and approaches used
3. Key results and findings
4. Stated limitations

Be concise but comprehensive. Organize by paper if multiple papers are provided.

Extracted Information:
"""

# Paper comparison prompt
COMPARISON_PROMPT = """
Papers to compare:
{papers}

Question: {query}

Compare these papers focusing on:
1. Core approaches and methodologies
2. Key differences in methods or findings
3. Strengths and limitations of each
4. How they relate to or build upon each other

Provide a structured comparison with clear distinctions.

Comparison:
"""


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with variables
    
    Args:
        template: Prompt template string
        **kwargs: Variables to fill in
    
    Returns:
        Formatted prompt
    """
    return template.format(**kwargs)


# Example usage
if __name__ == "__main__":
    # Test classification prompt
    query = "Compare BERT and GPT architectures"
    prompt = format_prompt(CLASSIFICATION_PROMPT, query=query)
    print(prompt)