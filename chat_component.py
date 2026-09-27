"""An in-page Streamlit chat input with project question suggestions."""

from pathlib import Path

import streamlit as st


_HERE = Path(__file__).parent

def suggested_chat_input(**kwargs):
    """Mount the input and read its assets on each rerun during local development."""
    component = st.components.v2.component(
        "machine_failure_suggested_chat_input",
        html="""
        <div class="question-entry">
          <div class="entry-row">
            <input id="question" type="text" role="combobox"
                   aria-label="Ask a question about this model"
                   aria-autocomplete="list" aria-controls="question-suggestions"
                   aria-expanded="false" autocomplete="off"
                   placeholder="Ask a question about this model" />
            <button id="send-question" type="button" aria-label="Send question">Send</button>
          </div>
          <div id="question-suggestions" role="listbox" hidden></div>
        </div>
        """,
        css=(_HERE / "chat_component.css").read_text(encoding="utf-8"),
        js=(_HERE / "chat_component.js").read_text(encoding="utf-8"),
    )
    return component(**kwargs)
