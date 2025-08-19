from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from src.graph_nodes import generate_query_or_respond, rewrite_question, generate_answer, grade_documents
from src.document_retriever import DocumentRetriever
from src.pydantic_models import GradeDocuments
from src.config import ConfigManager

# --- get the config ---
config = ConfigManager()._config


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
