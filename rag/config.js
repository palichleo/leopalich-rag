// rag/config.js
// Charge .env avant toute lecture de process.env : chaque point d'entrée (index.js, indexer.js,
// delete_collection.js) passe par ce module, quel que soit l'ordre des require.
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env'), quiet: true });

function int(name, fallback) {
  const v = parseInt(process.env[name] || '', 10);
  return Number.isFinite(v) ? v : fallback;
}

function list(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  return raw.split(',').map(s => s.trim()).filter(Boolean);
}

module.exports = {
  // Embeddings
  embedModel: process.env.EMBED_MODEL || 'Xenova/multilingual-e5-base',
  embedMaxTokens: int('EMBED_MAX_TOKENS', 0), // 0 = limite connue du modèle
  embedThreads: Math.max(1, int('EMBED_THREADS', 1)),
  modelCache: process.env.XENOVA_CACHE || '/opt/models',

  // Indexation
  indexBatch: int('INDEX_BATCH', 32),
  chunkMaxTokens: int('CHUNK_MAX_TOKENS', 256),
  chunkOverlapTokens: int('CHUNK_OVERLAP', 40),
  sentenceExplode: process.env.SENTENCE_EXPLODE === '1',

  // ChromaDB
  chromaHost: process.env.CHROMA_HOST || 'localhost',
  chromaPort: int('CHROMA_PORT', 8000),
  collectionName: process.env.CHROMA_COLLECTION || 'leoknowledge',

  // Génération
  groqModel: process.env.GROQ_MODEL || 'llama-3.1-8b-instant',
  groqTimeoutMs: int('GROQ_TIMEOUT_MS', 30000),

  // API
  port: int('PORT', 3000),
  allowedOrigins: list('ALLOWED_ORIGINS', ['https://leopalich.com', 'https://www.leopalich.com']),
  trustProxy: int('TRUST_PROXY', 1),
  promptMaxChars: int('PROMPT_MAX_CHARS', 500),
  ratePerMinute: int('RATE_LIMIT_PER_MINUTE', 5),
  dailyLimit: int('DAILY_LIMIT', 300),
};
