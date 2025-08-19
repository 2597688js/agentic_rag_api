from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from src.graph_nodes import generate_query_or_respond, rewrite_question, generate_answer, grade_documents
from src.document_retriever import DocumentRetriever
from src.pydantic_models import GradeDocuments

# --- get the config ---
try:
    from src.config import ConfigManager
    config = ConfigManager()._config
except Exception as e:
    print(f"Failed to load config in graph.py: {e}, using fallback")
    config = {
        'model_config': {
            'response_model': 'gpt-3.5-turbo',
            'grader_model': 'gpt-3.5-turbo',
            'temperature': 0.7
        },
        'prompts': {
            'GRADE_PROMPT': 'You are a document relevance grader. Your task is to determine if the retrieved documents are relevant to the user\'s question.\n\nQuestion: {question}\nRetrieved Context: {context}\n\nGrade the relevance using a binary score:\n- "yes" if the documents are relevant and can help answer the question\n- "no" if the documents are not relevant or insufficient\n\nBinary Score:',
            'REWRITE_PROMPT': 'You are a question rewriter. The user\'s question was not answered well by the retrieved documents.\n\nOriginal Question: {question}\n\nPlease rewrite the question to be more specific, clear, or focused.\n\nRewritten Question:',
            'GENERATE_PROMPT': 'You are an AI assistant that answers questions based on retrieved document content.\n\nQuestion: {question}\nRetrieved Context: {context}\n\nPlease provide a comprehensive answer based on the context provided. If the context doesn\'t contain enough information to answer the question completely, acknowledge what you can answer and what information is missing.\n\nAnswer:'
        }
    }


class MixRAGGraph(StateGraph):
    """A graph that combines the RAG workflow with a retrieval-based approach."""
    def __init__(self, retriever_tool: ToolNode):
        super().__init__(MessagesState)
        self.retriever_tool = retriever_tool
        self.workflow = self.create_workflow()
        # Note: max_iterations is defined but not currently used in the workflow
        # The loop prevention is handled in grade_documents function

    def generate_query_or_respond_with_tool(self, state):
        return generate_query_or_respond(self.retriever_tool, state)

    def create_workflow(self):
        workflow = StateGraph(MessagesState)

        # Define the nodes we will cycle between
        workflow.add_node("generate_query_or_respond", self.generate_query_or_respond_with_tool)
        workflow.add_node("retrieve", ToolNode([self.retriever_tool]))
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        # Edges taken after the `action` node is called.
        workflow.add_conditional_edges(
            "retrieve",
            # Assess agent decision
            grade_documents,
            {
                "generate_answer": "generate_answer",
                "rewrite_question": "rewrite_question",
            },
        )
        
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        # Compile
        graph = workflow.compile()

        return graph
    
    def display_graph(self):
        try:
            # Try to generate PNG
            png_data = self.workflow.get_graph().draw_mermaid_png()
            # Save to file
            with open("workflow.png", "wb") as f:
                f.write(png_data)
            print("Workflow saved as workflow.png")
        except Exception as e:
            print(f"Could not generate PNG: {e}")
            print("Falling back to Mermaid text:")
            print(self.workflow.get_graph().draw_mermaid())
