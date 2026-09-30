import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import BartTokenizer, BartForConditionalGeneration


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Text Paraphrasing",
    page_icon="🔄",
    layout="centered"
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = "Shubham22122/bart-paraphrasing-finetuned"

    tokenizer = BartTokenizer.from_pretrained(
        model_path
    )

    model = BartForConditionalGeneration.from_pretrained(
        model_path
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("🔄 AI Text Paraphrasing Tool")

st.write(
    "Enter a sentence below and the Full Fine-Tuned BART model "
    "will generate a paraphrased version."
)


sentence = st.text_area(
    "Enter your sentence:",
    placeholder="Ask anything...",
    height=120
)


# --------------------------------------------------
# Enter key support
# --------------------------------------------------

components.html(
    """
    <script>

    document.addEventListener("keydown", function(event) {

        if (event.key === "Enter" && !event.shiftKey) {

            const activeElement = document.activeElement;

            if (
                activeElement &&
                activeElement.tagName === "TEXTAREA"
            ) {

                event.preventDefault();

                const buttons = Array.from(
                    window.parent.document.querySelectorAll("button")
                );

                const generateButton = buttons.find(
                    button =>
                        button.innerText.includes(
                            "Generate Paraphrase"
                        )
                );

                if (generateButton) {
                    generateButton.click();
                }

            }

        }

    });

    </script>
    """,
    height=0
)


# --------------------------------------------------
# Generate paraphrase
# --------------------------------------------------

if st.button(
    "✨ Generate Paraphrase",
    use_container_width=True
):

    if not sentence.strip():

        st.warning(
            "Please enter a sentence first."
        )

    else:

        with st.spinner(
            "Generating paraphrase..."
        ):

            tokenizer, model, device = load_model()

            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                max_length=128,
                truncation=True
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                output = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )

            paraphrase = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )


        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        st.subheader(
            "Paraphrased Sentence"
        )

        st.success(paraphrase)


# --------------------------------------------------
# Information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: BART-base Full Fine-Tuning | Dataset: PAWS-X | "
    "Task: Automatic Text Paraphrasing"
)