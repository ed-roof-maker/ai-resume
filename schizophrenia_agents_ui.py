import gradio as gr


from schizophrenia_agents import agent, chat
import gradio as gr



# --------------------------------------------------
# White and blue theme
# --------------------------------------------------

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.sky,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    spacing_size=gr.themes.sizes.spacing_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill="#f3f7fd",
    body_background_fill_dark="#f3f7fd",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    body_text_color="#183153",
    body_text_color_dark="#183153",
    button_primary_background_fill="#1764c0",
    button_primary_background_fill_hover="#0d4f9e",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    border_color_primary="#d9e5f5",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

css = """
footer {
    display: none !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 24px 18px !important;
    font-family: Inter, Arial, sans-serif !important;
}

/* Hero banner */

.hero {
    background: linear-gradient(
        120deg,
        #0b3978 0%,
        #1764c0 55%,
        #42a5f5 100%
    );
    border-radius: 22px;
    padding: 32px 36px;
    color: white;
    box-shadow: 0 12px 30px rgba(18, 78, 153, 0.16);
    margin-bottom: 18px;
}

.hero h1 {
    color: white !important;
    font-size: 29px !important;
    font-weight: 750 !important;
    margin-bottom: 12px !important;
    line-height: 1.3;
}

.hero p {
    color: #e3f1ff !important;
    font-size: 15px !important;
    line-height: 1.7;
    margin: 0;
}

.hero-tag {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    color: white;
    padding: 6px 13px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 15px;
    letter-spacing: 0.4px;
}

/* Disclaimer */

.medical-notice {
    background: #eaf3ff;
    border: 1px solid #c9dfff;
    border-left: 4px solid #1764c0;
    border-radius: 12px;
    padding: 15px 19px;
    margin-bottom: 18px;
    color: #24466f;
    font-size: 13px;
    line-height: 1.7;
}

.medical-notice strong {
    color: #124a91;
}

/* Chat card */

#chat-card {
    background: white !important;
    border: 1px solid #e0eaf6 !important;
    border-radius: 20px !important;
    padding: 18px !important;
    box-shadow: 0 8px 30px rgba(30, 70, 130, 0.06);
}

/* Chat messages */

#chatbot {
    border: none !important;
    background: #ffffff !important;
}

#chatbot .message {
    border-radius: 16px !important;
    line-height: 1.65 !important;
    font-size: 14px !important;
    color: #ffffff !important;
}

#chatbot .message.user {
    background: #a0d9ef !important;
    color: #ffffff !important;
}

#chatbot .message.bot {
    background: #f0f6ff !important;
    color: #183153 !important;
    border: 1px solid #e0eafa !important;
}

/* Input */

#chat-card textarea {
    border-radius: 14px !important;
    font-size: 14px !important;
}

/* Example buttons */

#chat-card button {
    transition: all 0.2s ease;
}

#chat-card button:hover {
    transform: translateY(-1px);
}

/* Footer */

.app-footer {
    text-align: center;
    color: #7890ad;
    font-size: 12px;
    padding: 20px 0 8px;
}

/* Mobile */

@media (max-width: 600px) {
    .gradio-container {
        padding: 12px !important;
    }

    .hero {
        padding: 24px 20px;
        border-radius: 16px;
    }

    .hero h1 {
        font-size: 23px !important;
    }

    #chat-card {
        padding: 8px !important;
    }
}
"""

# --------------------------------------------------
# Example questions
# --------------------------------------------------

examples = [
    ["Can I just be on low-dose antipsychotics?"],
    ["Is it safe to live with schizophrenia without medication?"],
    ["Is Eleanor Longden living with schizophrenia without medication?"],
    ["Do antipsychotics cause weight gain?"],
    ["Can antipsychotics affect your liver?"],
    ["Do antipsychotics affect your libido?"],
    ["Can antipsychotics cause muscle twitching?"],
    ["What causes schizophrenia?"],
    ["Does cannabis cause schizophrenia?"],
    ["What is the relationship between stress and schizophrenia?"],
    ["Are people with schizophrenia dangerous?"],
    ["Do schizophrenics have multiple personalities?"],
    ["What support is available for people with schizophrenia in Australia?"],
    ["Can i get the disability pension if i have schizophrenia?"],
    ["Can i claim disabilty insurance if i have schizophrenia?"],
    ["Are there charities that help you if you have schizophrenia?"],
    ["Who can advocate for me if I do not consent to ECT?"],
    ["Besides antipsychotics, what can help manage schizophrenia?"],
    ["Is schizophrenia just god talking to me?"],
    ["Is schizophrenia a form of mind control, or electronic harassment?"],
    ["What is at the forefront of schizophrenia research, such as non-dopaminergic drug discovery?"],
    ["Should I employ someone with schizophrenia?"],
    ["Is it safe to employ someone with schizophrenia?"],
    ["Can I deny someone employment based on their disability?"],
]

# --------------------------------------------------
# Gradio application
# --------------------------------------------------

with gr.Blocks(
    title="Australian Schizophrenia Deep Research Demystifier",
) as demo:

    gr.HTML("""
        <div class="hero">
            <div class="hero-tag">
                AI-POWERED KNOWLEDGE EXPLORER
            </div>

            <h1>
                Australian Schizophrenia<br>
                Deep Research Demystifier BETA
            </h1>

            <p>
                Explore; research, evidence, and AI-debated
                topics about schizophrenia in Australia.
                Ask questions, examine different perspectives
                and understand the available evidence.
                Get a balanced debate based on evidence.
                Google does not do this.
            </p>
            <p> 
                Made by a schizophrenic. Visit my website<a href="https://edmonddendekker.com">here.</a>
            </p>
        </div>
    """)

    gr.HTML("""
        <div class="medical-notice">
            <strong>Important medical information</strong><br>
            This AI system provides educational information
            and may make mistakes. Its responses are not
            medical advice, diagnosis or treatment
            recommendations. Always consult a qualified
            healthcare professional before making decisions
            about medication or treatment.
        </div>
    """)

    thread_id = gr.State(value=None)

    with gr.Group(elem_id="chat-card"):

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[thread_id],
            additional_outputs=[thread_id],
            chatbot=gr.Chatbot(
                elem_id="chatbot",
                height=520,
                placeholder=(
                    "Welcome! Ask a question about "
                    "schizophrenia, research or support "
                    "services in Australia."
                ),
                show_label=False,
                layout="bubble",
            ),
            textbox=gr.Textbox(
                placeholder="Ask a question about schizophrenia...",
                container=True,
                scale=8,
                lines=2,
                max_lines=5,
                show_label=False,
                stop_btn=True,
                submit_btn=True,
            ),
            examples=examples,
            stop_btn=True,
            submit_btn=True,
        )

    gr.HTML("""
        <div class="app-footer">
            Australian Schizophrenia Knowledge Agent
            &nbsp;•&nbsp;
            AI-assisted research
            &nbsp;•&nbsp;
            Educational use only
        </div>
    """)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        root_path='/schizophrenia',
        show_error=False,
        theme=theme,
        css=css,
    )