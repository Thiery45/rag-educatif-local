import streamlit as st
from rag import agent_router

st.set_page_config(
    page_title="Assistant RAG Éducatif",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Assistant RAG Éducatif")
st.write(
    "Plateforme RAG éducative locale capable d'interroger les PDF placés dans le dossier `data/`. "
    "L'assistant utilise FAISS, des embeddings HuggingFace, Ollama et un agent simple avec outils."
)

question = st.text_input(
    "Ta question :",
    placeholder="Exemple : Quels sont les paramètres clés du RAG ?"
)

top_k = st.slider(
    "Nombre de passages récupérés (top-k)",
    min_value=1,
    max_value=8,
    value=3
)

if st.button("Interroger les documents"):
    if question.strip() == "":
        st.warning("Veuillez saisir une question.")
    else:
        with st.spinner("L'agent analyse la demande..."):
            mode, result = agent_router(question, top_k=top_k)

        if mode == "rag":
            st.subheader("Réponse")
            st.write(result["answer"])

            st.subheader("Sources utilisées")
            for s in result["sources"]:
                st.write(
                    f"📄 **{s['source']}** — page {s['page']} "
                    f"— score relatif : **{s['relevance']:.2f}%**"
                )

        elif mode == "sources":
            st.subheader("Sources disponibles")
            for source in result:
                st.write(f"📄 {source}")

        elif mode == "summary":
            st.subheader("Résumé du corpus")
            st.write(result)

st.divider()

st.caption(
    "Architecture : PDF → chunks → embeddings → index FAISS → retriever → prompt augmenté → LLM local Ollama → réponse sourcée."
)