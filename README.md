Voici un **README complet** prêt à mettre dans ton dépôt GitHub. Tu peux directement créer un fichier `README.md` et y coller ce contenu.

---

# Assistant RAG Éducatif Local

## Description

Ce projet implémente un **système RAG (Retrieval-Augmented Generation) éducatif** entièrement local.
Il permet d’interroger n’importe quel corpus de documents PDF et de générer des réponses **contexte-dépendantes**, en citant les sources utilisées pour chaque réponse.

Le système est générique : tu peux remplacer les PDF du dossier `data/` par tes propres documents, et le RAG s’adaptera automatiquement.

---

## Objectifs

* Fournir des réponses précises basées uniquement sur les documents fournis.
* Afficher les sources exactes (nom du document + page).
* Permettre une utilisation entièrement locale, sans dépendances cloud ou API payante.
* Expérimenter avec l’orchestration simple d’agents (RAG, résumé, liste de sources).
* Démontrer l’architecture complète d’un assistant IA fiable pour l’éducation ou l’entreprise.

---

## Architecture du projet

```text
PDF (data/)
 ↓
PyPDFLoader (lecture texte + métadonnées)
 ↓
Découpage en chunks (chunk_size=800, overlap=150)
 ↓
Embeddings (all-MiniLM-L6-v2)
 ↓
FAISS (base vectorielle locale)
 ↓
Recherche Top-K = 3
 ↓
Prompt augmenté
 ↓
LLM local (Llama 3.2 via Ollama)
 ↓
Réponse + Sources + Scores relatifs
```

### Agent / Outils

* **Recherche documentaire** : retourne une réponse basée sur les documents.
* **Résumé du corpus** : résume tous les documents indexés.
* **Consultation des sources** : affiche tous les PDF et pages disponibles.

---

## Contenu du projet

```text
rag-project/
├── data/                # Dossier contenant les PDF à indexer
├── vectorstore/         # Index FAISS généré
├── ingest.py            # Lecture PDF, chunking, embeddings, construction FAISS
├── rag.py               # Recherche, LLM, agent et gestion des sources
├── app.py               # Interface Streamlit
├── requirements.txt     # Bibliothèques Python nécessaires
└── README.md
```

---

## Installation

1. Cloner le projet :

```bash
git clone <URL_DU_PROJET>
cd rag-project
```

2. Créer un environnement virtuel Python (recommandé Python 3.12) :

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Mac/Linux
```

3. Installer les dépendances :

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Installer Ollama et télécharger le modèle Llama 3.2 :

```bash
# Installer Ollama depuis https://ollama.com/download
ollama pull llama3.2
```

---

## Utilisation

### Préparer les documents

Placer les PDF à interroger dans le dossier `data/`.

### Construire l’index FAISS

```bash
python ingest.py
```

### Lancer l’interface web

```bash
streamlit run app.py
```

* Poser des questions sur les documents.
* Obtenir les réponses avec les sources et le score relatif de pertinence.
* Options : résumé du corpus, consultation des sources disponibles.

### Utilisation CLI (optionnel)

```bash
python rag.py
```

* Pose directement une question dans le terminal.
* Obtenir la réponse et les sources.

---

## Paramètres clés

* **Chunk size** : 800 tokens
* **Overlap** : 150 tokens
* **Top-K** : 3
* **Embedding** : `all-MiniLM-L6-v2`
* **LLM** : `Llama 3.2` via Ollama
* **Température** : 0.2 (réponses cohérentes, peu créatives)

Ces paramètres ont été optimisés de manière **itérative**, en testant différentes valeurs pour maximiser la pertinence, la qualité des sources et réduire les hallucinations.

---

## Pistes d’amélioration

* Supporter l’upload de PDF via l’interface.
* Ajouter DOCX/TXT.
* Recherche hybride FAISS + BM25.
* Re-ranking automatique des chunks.
* Historique conversationnel.
* Agents multi-étapes complexes (LangGraph).
* Évaluation automatique des réponses pour benchmarking.

---

## Licence

MIT License – utilisation éducative et personnelle autorisée.
## activer variable environnement
.venv\Scripts\activate

## installer bibliotheque
pip install -r requirements.txt

## pour lancer le projet
python ingest.py
streamlit run app.py
## 