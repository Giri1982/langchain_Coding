import gradio as gr

from chat_service import run_chat_with_audit


def launch_gradio() -> None:
    app = gr.Interface(
        fn=run_chat_with_audit,
        inputs=[
            gr.Textbox(
                label="Your Message",
                placeholder="Try: What is the weather in Chennai?",
                lines=3,
            ),
            gr.Checkbox(label="Show Audit Log", value=False),
            gr.Checkbox(label="Enable Memory", value=False),
            gr.State(value={"thread_id": None, "memory_history": []}),
        ],
        outputs=[
            gr.Textbox(label="Assistant Response", lines=8),
            gr.Textbox(label="Audit Log", lines=10),
            gr.Textbox(label="Stored Memory", lines=10),
            gr.State(),
        ],
        title="LangChain Weather Assistant",
        description=(
            "Ask anything. For weather questions, the model automatically calls the weather tool. "
            "Enable 'Show Audit Log' to see whether a tool was called. "
            "Enable 'Enable Memory' to remember context within the current local session."
        ),
    )
    app.launch()