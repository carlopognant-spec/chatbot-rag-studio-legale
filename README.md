# Chatbot RAG per Studio Legale

Assistente informativo basato su RAG (Retrieval-Augmented Generation) per lo **Studio Legale Rossi & Partners**. Risponde esclusivamente sui documenti interni, citando sempre la fonte.

## Stack tecnologico

- **Python 3.12** + **Streamlit** (interfaccia chat)
- **Google Gemini API** (embedding + generazione)
- **ChromaDB** (vector database in container Docker separato)
- **Docker / Docker Compose** per l'esecuzione containerizzata

## Documenti indicizzati

| File | Contenuto |
|------|-----------|
| `documenti/clausola_contrattuale.txt` | Clausole contrattuali tipo |
| `documenti/informativa_privacy.txt` | Informativa privacy GDPR |
| `documenti/estratto_regolamento.txt` | Regolamento interno dello studio |

## Prerequisiti

- Python 3.12+
- Docker e Docker Compose (per esecuzione containerizzata)
- Chiave API Google Gemini ([Google AI Studio](https://aistudio.google.com/apikey))

## Configurazione chiave API

Copia il template e inserisci la tua chiave:

```bash
cp .env.example .env   # Linux/macOS
copy .env.example .env # Windows
```

Modifica `.env` (non committarlo):

```env
GEMINI_API_KEY=la_tua_chiave_api_qui
```

## Esecuzione locale (senza Docker)

### 1. Avvia ChromaDB

```bash
docker run -d --name chromadb -p 8000:8000 -v chroma-data:/chroma/chroma -e IS_PERSISTENT=TRUE chromadb/chroma:latest
```

### 2. Installa dipendenze

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Imposta variabili d'ambiente

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "la_tua_chiave"
$env:CHROMADB_HOST = "localhost"
$env:CHROMADB_PORT = "8000"

# Linux/macOS
export GEMINI_API_KEY="la_tua_chiave"
export CHROMADB_HOST="localhost"
export CHROMADB_PORT="8000"
```

### 4. Avvia l'app Streamlit

```bash
streamlit run app.py --server.port=8080 --server.address=0.0.0.0
```

Apri il browser su [http://localhost:8080](http://localhost:8080).

## Esecuzione con Docker Compose

```bash
# Assicurati che .env contenga GEMINI_API_KEY
docker compose up --build
```

Servizi:
- **chromadb** → porta 8000 (persistenza su volume `chroma-data`)
- **app** → porta 8080 (Streamlit)

## Esempio domanda / risposta

**Domanda:** Entro quanti giorni posso recedere dal contratto dopo la sottoscrizione?

**Risposta attesa:**

> Il Cliente può esercitare il diritto di recesso entro 14 giorni dalla sottoscrizione del contratto, senza obbligo di motivazione e senza penali.
>
> Fonte: clausola_contrattuale.txt

**Domanda fuori contesto:** Chi è il sindaco di Torino?

**Risposta attesa:**

> Non ho trovato questa informazione nei documenti disponibili. Non è un parere legale: ti consiglio di verificare con lo studio.

## Valutazione (golden dataset)

```bash
# Con ChromaDB attivo e GEMINI_API_KEY impostata
python eval.py
```

Lo script esegue 20 domande di test da `golden_dataset.json` e stampa accuratezza, tabella riepilogativa e dettaglio fallimenti.

## Deploy

### Google Cloud Run

1. Build e push dell'immagine su Artifact Registry / Container Registry
2. Deploy del servizio app su Cloud Run (porta 8080)
3. Deploy di ChromaDB su Cloud Run separato o su GCE/GKE con volume persistente
4. Imposta `GEMINI_API_KEY` come secret / variabile d'ambiente
5. Configura `CHROMADB_HOST` con l'URL del servizio ChromaDB

### Hugging Face Spaces

1. Crea uno Space Docker
2. Adatta `compose.yaml` o usa un Dockerfile unico
3. Aggiungi `GEMINI_API_KEY` nei Secrets dello Space
4. Espone la porta 8080

## Struttura progetto

```
Chatbot_RAG/
├── documenti/              # Documenti sorgente
├── app.py                  # Interfaccia Streamlit
├── rag.py                  # Pipeline RAG
├── eval.py                 # Script di valutazione
├── golden_dataset.json     # Dataset di test
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .env.example            # Template variabili d'ambiente
├── .env                    # Chiave API (non committare)
├── .gitignore
└── .dockerignore
```

## Note sui modelli Gemini

I modelli predefiniti sono configurabili via variabili d'ambiente:

| Variabile | Default | Uso |
|-----------|---------|-----|
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding per retrieval |
| `GEMINI_GENERATION_MODEL` | `gemini-2.5-flash` | Generazione risposta |

Verifica i nomi aggiornati nella [documentazione Gemini](https://ai.google.dev/gemini-api/docs/models) se ricevi errori di modello non disponibile.

## Licenza

Progetto didattico — ITS Academy Piemonte, Gen AI Specialist.
