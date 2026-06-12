from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

VECTORSTORE_DIR = "vectorstore"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

def format_sources(docs_with_scores):
    sources = []

    if not docs_with_scores:
        return sources

    max_score = max(score for _, score in docs_with_scores)

    for doc, score in docs_with_scores:
        source = doc.metadata.get("source", "Source inconnue")
        page = doc.metadata.get("page", "Page inconnue")

        if max_score == 0:
            relevance = 100
        else:
            relevance = round(float((1 - (score / max_score)) * 100), 2)

        sources.append({
            "source": source,
            "page": page,
            "relevance": relevance
        })

    return sources


def ask_rag(question, top_k=3):
    vectorstore = load_vectorstore()

    docs_with_scores = vectorstore.similarity_search_with_score(
        question,
        k=top_k
    )

    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source')} - page {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc, score in docs_with_scores
        ]
    )

    prompt = f"""
Tu es un assistant documentaire pédagogique.

Règles obligatoires :
1. Réponds uniquement à partir du contexte fourni.
2. N'utilise aucune connaissance externe.
3. Si l'information n'est pas présente dans le contexte, réponds exactement :
"Je ne trouve pas cette information dans les documents fournis."
4. Ne donne pas de définition inventée.
5. Cite les sources utilisées à la fin de la réponse.

Question :
{question}

Contexte :
{context}

Réponse :
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]
    sources = format_sources(docs_with_scores)

    return answer, sources


def list_available_sources():
    vectorstore = load_vectorstore()
    docs = vectorstore.docstore._dict.values()

    sources = sorted(
        set(
            f"{doc.metadata.get('source', 'Source inconnue')} - page {doc.metadata.get('page', 'Page inconnue')}"
            for doc in docs
        )
    )

    return sources


def summarize_documents():
    sources = list_available_sources()

    prompt = f"""
Tu es un assistant documentaire.

Voici la liste des documents et pages disponibles :
{sources[:30]}

Fais un court résumé du corpus documentaire.
Ne prétends pas connaître le contenu complet si seules les sources sont listées.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


def agent_router(user_input, top_k=3):
    text = user_input.lower()

    if "source" in text or "document" in text or "pdf" in text:
        return "sources", list_available_sources()

    if "résume" in text or "resume" in text or "résumé" in text:
        return "summary", summarize_documents()

    answer, sources = ask_rag(user_input, top_k=top_k)
    return "rag", {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    question = input("Pose ta question : ")
    mode, result = agent_router(question)

    if mode == "rag":
        print("\nRéponse :")
        print(result["answer"])

        print("\nSources :")
        for s in result["sources"]:
            print(f"- {s['source']} - page {s['page']} | pertinence : {s['relevance']}%")

    elif mode == "sources":
        print("\nSources disponibles :")
        for source in result:
            print("-", source)

    elif mode == "summary":
        print("\nRésumé :")
        print(result)