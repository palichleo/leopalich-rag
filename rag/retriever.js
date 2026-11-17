// rag/retriever.js
const config = require('./config');
const { ChromaClient } = require('chromadb');
const { embedQueries, embedPassages } = require('./embedder');

const client = new ChromaClient({ host: config.chromaHost, port: config.chromaPort, ssl: false });
let collection = null;

// Les embeddings sont toujours calculés ici et passés à Chroma. Avec `embeddingFunction: null`,
// chromadb 3 tente de charger @chroma-core/default-embed (non installé) et la création de la
// collection échoue ; une fonction déclarée mais jamais appelée l'en empêche.
const externalEmbeddings = {
  async generate() {
    throw new Error('Embeddings calculés par rag/embedder.js : passer `embeddings` explicitement.');
  },
};

async function initCollection() {
  if (collection) return collection;
  const name = config.collectionName;
  try {
    collection = await client.getCollection({ name, embeddingFunction: externalEmbeddings });
  } catch {
    collection = await client.createCollection({
      name,
      embeddingFunction: externalEmbeddings,
      metadata: { 'hnsw:space': 'cosine' },
    });
  }
  return collection;
}

// Supprime la collection (si elle existe) et en recrée une vide.
async function resetCollection() {
  try {
    await client.deleteCollection({ name: config.collectionName });
  } catch {
    // collection absente : rien à supprimer
  }
  collection = null;
  return initCollection();
}

// ---------------- utils
function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function norm(a) { return Math.sqrt(dot(a, a)); }
function cosine(a, b) { return dot(a, b) / (norm(a) * norm(b) + 1e-10); }

// Mots vides FR (petit ensemble suffisant pour nos requêtes)
const STOP = new Set('je tu il elle nous vous ils elles de du des le la les un une et ou en au aux sur pour par avec sans sous chez dans que qui quoi dont ou est sont ete etais etait etre avoir ai as a avons avez ont plus moins tres ne pas ni mais donc or car quel quelle quels quelles'.split(' '));
function normalize(str) {
  return (str || '')
    .toLowerCase()
    .normalize('NFKD').replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9\s-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}
function tokens(str) {
  return normalize(str).split(' ').filter(t => t && !STOP.has(t));
}
// Part des mots de la question présents dans l'extrait (et non Jaccard : un extrait long
// ne doit pas être pénalisé parce qu'il contient d'autres mots).
function coverage(queryTokens, docTokens) {
  const Q = new Set(queryTokens);
  if (!Q.size) return 0;
  const D = new Set(docTokens);
  let hit = 0;
  for (const t of Q) if (D.has(t)) hit++;
  return hit / Q.size;
}
function yearsIn(text) {
  return (text.match(/\b(19|20)\d{2}\b/g) || []).map(Number);
}
const RECENCY_WORDS = /\b(recent|recents|recente|recentes|dernier|derniere|derniers|dernieres|actuel|actuelle|actuellement|aujourd hui|maintenant|en ce moment|en cours)\b/;

// Bonus temporel, appliqué seulement quand la question porte sur le temps :
// - la question cite une année -> bonus aux extraits de cette année ;
// - la question demande du récent -> petit bonus aux extraits récents.
function temporalBonus(query, year, yearNow) {
  if (!year) return 0;
  const asked = yearsIn(query);
  if (asked.length) return asked.includes(year) ? 0.1 : 0;
  if (RECENCY_WORDS.test(normalize(query))) {
    return 0.05 * Math.max(0, 1 - Math.min(5, Math.abs(yearNow - year)) / 5);
  }
  return 0;
}

// ---------------- MMR (diversité)
function mmr(queryEmb, candEmbs, lambda = 0.65, topK = 6) {
  const n = candEmbs.length;
  const selected = [];
  const used = new Set();
  const simToQuery = candEmbs.map(e => cosine(queryEmb, e));
  while (selected.length < Math.min(topK, n)) {
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      if (used.has(i)) continue;
      let div = 0;
      for (const j of selected) div = Math.max(div, cosine(candEmbs[i], candEmbs[j]));
      const score = lambda * simToQuery[i] - (1 - lambda) * div;
      if (score > bestScore) { bestScore = score; best = i; }
    }
    used.add(best);
    selected.push(best);
  }
  return selected;
}

// ---------------- Variantes de la question, sans LLM
const SYNONYMS = [
  ['parcours', 'experience', 'cv', 'formation'],
  ['projet', 'realisations', 'travaux', 'portfolio'],
  ['intelligence artificielle', 'ia', 'machine learning', 'ml'],
  ['traitement de donnees', 'analyse de donnees', 'data', 'statistiques'],
  ['freelance', 'independant', 'consultant'],
];

function expandQueriesLocal(query, n = 3) {
  const q = normalize(query);
  const variants = new Set([q]);

  // 1) mots-clés seuls
  const toks = tokens(query);
  if (toks.length) variants.add(toks.join(' '));

  // 2) synonymes du domaine, sur des mots entiers et à chaque occurrence
  for (const group of SYNONYMS) {
    for (const w of group) {
      const re = new RegExp(`\\b${w}s?\\b`, 'g');
      if (!re.test(q)) continue;
      for (const v of group) if (v !== w) variants.add(q.replace(re, v));
    }
  }

  return Array.from(variants).slice(0, n + 1); // question + n variantes
}

// ---------------- ajout (embeddings calculés ici, identifiants stables)
async function addDocuments(items) {
  if (!items || items.length === 0) return;
  const col = await initCollection();
  const embeddings = await embedPassages(items.map(x => x.text));
  await col.upsert({
    ids: items.map(x => x.id),
    documents: items.map(x => x.text),
    metadatas: items.map(x => x.metadata || {}),
    embeddings,
  });
}

/**
 * retrieveRelevant(query, kFinal, kInitial)
 * - variantes locales de la question -> union des candidats
 * - reclassement : similarité cosinus + couverture lexicale + bonus temporel si demandé
 * - MMR -> diversité
 */
async function retrieveRelevant(query, kFinal = 6, kInitial = 64) {
  const col = await initCollection();

  const queries = expandQueriesLocal(query, 3);
  const qTokens = tokens(query);
  const qEmbs = await embedQueries(queries);

  const candidates = new Map();
  for (const qEmb of qEmbs) {
    const res = await col.query({
      queryEmbeddings: [qEmb],
      nResults: kInitial,
      include: ['documents', 'metadatas', 'embeddings'],
    });
    const ids = res.ids?.[0] || [];
    const docs = res.documents?.[0] || [];
    const metas = res.metadatas?.[0] || [];
    const embs = res.embeddings?.[0] || [];
    for (let i = 0; i < ids.length; i++) {
      if (!candidates.has(ids[i])) {
        candidates.set(ids[i], { id: ids[i], text: docs[i], meta: metas[i] || {}, emb: embs[i] });
      }
    }
  }
  const pool = Array.from(candidates.values());
  if (pool.length === 0) return [];

  const yearNow = new Date().getFullYear();
  const qEmbMain = qEmbs[0];
  const ranked = pool.map((d) => {
    const cos = cosine(qEmbMain, d.emb);
    const lex = coverage(qTokens, tokens(d.text));
    const score = 0.85 * cos + 0.15 * lex + temporalBonus(query, d.meta.year, yearNow);
    return { ...d, score };
  }).sort((a, b) => b.score - a.score);

  const topForMMR = ranked.slice(0, Math.max(kFinal * 4, 24));
  const idxs = mmr(qEmbMain, topForMMR.map(x => x.emb), 0.65, Math.min(kFinal * 2, topForMMR.length));
  const final = idxs.map(i => topForMMR[i]).sort((a, b) => b.score - a.score).slice(0, kFinal);

  return final.map((x, i) => ({
    i,
    id: x.id,
    source: x.meta.source || 'inconnu',
    score: +x.score.toFixed(4),
    text: x.text,
  }));
}

module.exports = {
  initCollection, resetCollection, addDocuments, retrieveRelevant,
  // exportés pour les tests
  expandQueriesLocal, temporalBonus, coverage,
};
