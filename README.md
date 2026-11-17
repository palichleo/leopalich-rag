# API RAG de leopalich.com

Petit serveur qui répondait aux questions des visiteurs du site à partir de mes propres documents
(CV, fiches de projets). Projet personnel de l'été 2025 ; l'interface a depuis été retirée du site.

## Chaîne de traitement

1. **Découpage** (`rag/indexer.js`) : chaque fichier Markdown de `knowledges/` est coupé par
   titres, puis en phrases. Le premier titre du fichier (le nom du projet) et le chemin de titres
   (`H1 > H2 > H3`) restent en tête de chaque extrait, avec l'année mentionnée. Les phrases sont
   regroupées jusqu'à 256 tokens, comptés avec le tokenizer du modèle d'embedding, et chaque
   extrait reprend les dernières phrases du précédent (40 tokens au plus).
2. **Embeddings** (`rag/embedder.js`) : `Xenova/multilingual-e5-base` quantifié, exécuté sur le
   processeur avec `@xenova/transformers`, avec les préfixes `query: ` et `passage: ` attendus par
   e5. Aucun appel externe pour cette étape.
3. **Stockage** (`rag/retriever.js`) : ChromaDB dans Docker. Chaque extrait a un identifiant
   stable (`fichier#numéro`) ; une réindexation vide la collection puis la reconstruit.
4. **Recherche** : la question est déclinée en variantes locales (mots-clés seuls, synonymes du
   domaine), sans appel à un modèle. Les 64 voisins de chaque variante sont réunis, puis reclassés
   par la similarité cosinus (0,85) et la part des mots de la question présents dans l'extrait
   (0,15). Un bonus temporel s'ajoute seulement si la question cite une année ou demande du
   récent. Une sélection MMR diversifie les extraits ; 10 sont gardés.
5. **Génération** (`index.js`) : les extraits numérotés sont envoyés à `llama-3.1-8b-instant` chez
   Groq, température 0, avec consigne de répondre uniquement depuis les extraits ou de répondre
   « Je n'ai pas accès à cette information ». La réponse est renvoyée en flux continu, débarrassée
   du gras Markdown.

## Protection de l'API

Chaque question coûte un appel Groq. L'API accepte seulement les origines de `ALLOWED_ORIGINS`,
limite chaque adresse IP à 5 questions par minute et l'ensemble à 300 par jour, refuse les
questions de plus de 500 caractères, coupe la génération si le visiteur ferme la page et ne
renvoie jamais le détail d'une erreur. Tous ces seuils se règlent dans `.env`.

Derrière nginx, `TRUST_PROXY=1` permet de lire l'adresse du visiteur dans `X-Forwarded-For`.
Sans proxy, mettre `TRUST_PROXY=0`.

## Lancer

```sh
npm install
cp .env.example .env        # puis renseigner GROQ_API_KEY
npm run chroma              # ChromaDB dans Docker, port 8000
npm run index               # vide la collection et indexe knowledges/
npm start                   # API sur le port 3000
npm test                    # tests du découpage, de la recherche et du flux, sans réseau
```

```sh
curl -X POST localhost:3000/ask -H "Content-Type: application/json" -d '{"prompt":"Quels projets ?"}'
```

Changer `EMBED_MODEL` impose de réindexer : les vecteurs de deux modèles ne se comparent pas.