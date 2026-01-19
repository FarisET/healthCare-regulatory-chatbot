"""
Hospital Audit Agent - Chatbot Interface
A modern Gradio-based interface for the Ask-Audit Agent
Compatible with Gradio 6.0+
"""

import gradio as gr
import json
from typing import List, Tuple
from langchain_core.messages import HumanMessage

# Import your agent and utilities from the main module
# Adjust the import path based on your file structure
try:
    from chatbot6 import rag_agent, _format_department_analysis_md
except ImportError:
    print("Warning: Could not import agent. Make sure chatbot6.py is accessible.")
    rag_agent = None
    _format_department_analysis_md = None


class AuditChatbot:
    """Wrapper class for the audit agent chatbot"""
    
    def __init__(self):
        self.conversation_history = []
        self.agent = rag_agent
    
    def format_analysis_output(self, content: str) -> str:
            """
            Process agent output and apply formatting based on tool output structure.
            Many tools return {"analysis": {...}, "markdown": "..."} format.
            """
            # Check if content contains JSON with pre-formatted markdown
            try:
                # simple heuristic to find json
                if content.strip().startswith('{'):
                    data = json.loads(content)
                    
                    # Priority 1: Tool has already provided formatted markdown
                    if 'markdown' in data and isinstance(data['markdown'], str):
                        return data['markdown']
                    
                    # Priority 2: Use _format_department_analysis_md for legacy format
                    # FIX: We now strictly check if data['analysis'] is a DICT
                    if 'analysis' in data:
                        analysis_data = data['analysis']
                        
                        # Only proceed if analysis_data is a dictionary
                        if isinstance(analysis_data, dict):
                            if 'department' in analysis_data:
                                if _format_department_analysis_md:
                                    return _format_department_analysis_md(analysis_data)
                            
                            # Priority 3: Manual formatting if the helper didn't run
                            lines = []
                            if 'department' in analysis_data:
                                lines.append(f"## Department Analysis: {analysis_data['department']}\n")
                            if 'gap_score' in analysis_data:
                                lines.append(f"**Gap Score:** {analysis_data['gap_score']}/100\n")
                            if 'note' in analysis_data:
                                lines.append(f"**Summary:** {analysis_data['note']}\n")
                            if 'linked_clauses_count' in analysis_data:
                                lines.append(f"- Linked clauses: {analysis_data['linked_clauses_count']}")
                            if 'strong_suggestions_count' in analysis_data:
                                lines.append(f"- Strong suggestions: {analysis_data['strong_suggestions_count']}")
                            if 'weak_suggestions_count' in analysis_data:
                                lines.append(f"- Weak suggestions: {analysis_data['weak_suggestions_count']}")
                            
                            if lines:
                                return "\n".join(lines)
                    
                    # Priority 4: Check for error messages
                    if 'error' in data:
                        return f"⚠️ **Error:** {data['error']}"
                        
            except json.JSONDecodeError:
                pass
            except Exception as e:
                # Fallback for any other formatting errors
                return f"{content}\n\n*(Formatting Warning: {str(e)})*"
            
            return content
    
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """
        Process user message and return updated history
        """
        if not self.agent:
            return history + [(message, "⚠️ Agent not initialized. Please check your imports.")], ""
        
        if not message.strip():
            return history, ""
        
        try:
            # Create message for agent
            messages = [HumanMessage(content=message)]
            
            # Invoke agent
            result = self.agent.invoke({"messages": messages})
            
            # Extract response
            out_msgs = result.get('messages', [])
            if not out_msgs:
                response = "❌ No response from agent."
            else:
                final = out_msgs[-1]
                response = final.content if hasattr(final, "content") else str(final)
                
                # Apply formatting if needed
                response = self.format_analysis_output(response)
            
            # Update history
            self.conversation_history.append((message, response))
            history.append((message, response))
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error processing request: {str(e)}"
            history.append((message, error_msg))
            return history, ""
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        return []


def create_chatbot_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize chatbot
    bot = AuditChatbot()
    
    # Example queries
    example_queries = [
        "Analyze the Emergency Department checklist and suggest improvements",
        "What are the gaps regarding hand hygeine audit compared to SHCC framework?",
        "Does my Emergency Department checklist cover high alert medication according to JCI standards",
        "Find clauses related to infection control accross SHCC and JCI",
        "What requirements are mentioned in JCI and SHCC for patient handover?",
    ]
    
    # Create interface
    with gr.Blocks(title="Ask-Audit Agent") as interface:
        
        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                <h1>🏥 Ask-Audit Agent</h1>
                <p>Expert HealthCare Quality Management & Regulatory Compliance Assistant</p>
            </div>
        """)
        
        # Main chat interface
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    avatar_images=(None, "🤖"),
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about audits, checklists, gaps, or compliance...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Conversation", size="sm")
            
            # Sidebar with info and examples
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Capabilities")
                gr.Markdown("""
                - **Department Analysis**: Review checklists and find gaps
                - **Framework Comparison**: Compare against NICE, WHO, CQC
                - **Concept Search**: Find related compliance clauses
                - **Gap Analysis**: Identify audit deficiencies
                - **Smart Recommendations**: Data-driven suggestions
                """)
                
                gr.Markdown("### 💡 Example Questions")
                example_buttons = []
                for example in example_queries:
                    btn = gr.Button(example, size="sm", variant="secondary")
                    example_buttons.append((btn, example))
        
        # Help section
        with gr.Accordion("ℹ️ How to Use", open=False):
            gr.Markdown("""
            ### Query Types Supported:
            
            1. **Department Analysis**
               - "Analyze the [Department] checklist"
               - Provides linked clauses, strong suggestions, and weak suggestions
            
            2. **Gap Analysis**
               - "What are the gaps in [department/audit] compared to [framework]?"
               - Identifies missing compliance requirements
            
            3. **Framework Comparison**
               - "Compare [framework A] with [framework B]"
               - Shows overlaps and unique clauses
            
            4. **Concept Search**
               - "Find clauses about [topic]"
               - Searches knowledge base by medical concept
            
            5. **General Guidance**
               - Ask open-ended questions about audit best practices
            
            ### Tips:
            - Be specific about department names (e.g., "Emergency Department", "ICU")
            - Mention frameworks explicitly (NICE, WHO, CQC, etc.)
            - Ask follow-up questions to dive deeper
            - Use the example queries to get started
            """)
        
        # Event handlers
        def submit_message(message, history):
            return bot.process_message(message, history)
        
        def use_example(example_text, history):
            return bot.process_message(example_text, history)
        
        # Connect events
        msg.submit(submit_message, [msg, chatbot], [chatbot, msg])
        submit_btn.click(submit_message, [msg, chatbot], [chatbot, msg])
        clear_btn.click(bot.clear_conversation, None, chatbot)
        
        # Connect example buttons
        for btn, example_text in example_buttons:
            btn.click(
                lambda ex=example_text: bot.process_message(ex, []),
                None,
                chatbot
            )
    
    return interface


def launch_chatbot(share=False, server_port=7860):
    """
    Launch the chatbot interface
    
    Args:
        share: Whether to create a public link (default: False)
        server_port: Port to run the server on (default: 7860)
    """
    interface = create_chatbot_interface()
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="localhost",
        show_error=True
    )


if __name__ == "__main__":
    # Launch with default settings
    # Set share=True to create a public link
    launch_chatbot(share=False, server_port=7860)