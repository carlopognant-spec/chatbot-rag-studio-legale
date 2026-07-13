"""
Interfaccia chat Streamlit per il chatbot RAG dello studio legale.
"""

import os

import streamlit as st

from rag import ask, index_documents

st.set_page_config(
    page_title="Chatbot RAG — Studio Legale",
    page_icon="⚖️",
    layout="centered",
)


@st.cache_resource
def build_index() -> int:
    """
    Costruisce l'indice una sola volta per processo Streamlit.
    cache_resource evita di re-indicizzare ad ogni rerun dell'interfaccia.
    """
    return index_documents()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.title("⚖️ Chatbot RAG — Studio Legale Rossi & Partners")
    st.caption(
        "Assistente informativo basato sui documenti interni. "
        "Non sostituisce una consulenza legale."
    )

    if "GEMINI_API_KEY" not in os.environ:
        st.error(
            "Variabile d'ambiente GEMINI_API_KEY non impostata. "
            "Configurala nel file .env o nelle variabili di sistema."
        )
        st.stop()

    init_session_state()

    with st.spinner("Indicizzazione documenti in corso..."):
        try:
            chunk_count = build_index()
        except Exception as exc:
            st.error(f"Errore durante l'indicizzazione: {exc}")
            st.stop()

    st.sidebar.success(f"Indice pronto: {chunk_count} chunk indicizzati.")
    st.sidebar.markdown(
        "**Documenti disponibili:**\n"
        "- clausola_contrattuale.txt\n"
        "- informativa_privacy.txt\n"
        "- estratto_regolamento.txt"
    )

    if st.sidebar.button("Svuota cronologia"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"Fonti: {', '.join(message['sources'])}")

    if prompt := st.chat_input("Scrivi la tua domanda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sto consultando i documenti..."):
                try:
                    result = ask(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as exc:
                    answer = f"Si è verificato un errore: {exc}"
                    sources = []

            st.markdown(answer)
            if sources:
                st.caption(f"Fonti: {', '.join(sources)}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
