// rag/embedder.js
const config = require('./config');

// onnxruntime-web doit être configuré avant le chargement de transformers.
const ort = require('onnxruntime-web');
ort.env.wasm.proxy = false;

const { pipeline, AutoTokenizer, env } = require('@xenova/transformers');

// Sous Node, onnxruntime-web ne sait pas lancer ses workers (ERR_WORKER_PATH) : au-delà d'un
// thread, le chargement du modèle échoue puis se bloque. L'ancien réglage (jusqu'à 4 threads)
// ne fonctionnait que sur une machine à un cœur.
env.backends.onnx.wasm.numThreads = config.embedThreads;
env.backends.onnx.wasm.proxy = false;
env.allowRemoteModels = true;
env.localModelPath = config.modelCache;

const MODEL_ID = config.embedModel;

// Longueur de séquence sur laquelle chaque modèle a été entraîné. Au-delà, le texte est tronqué
// (ou, pour MiniLM, encodé hors de son domaine d'entraînement).
const KNOWN_LIMITS = [
  [/multilingual-e5/i, 512],
  [/MiniLM-L12/i, 128],
];

// Les modèles e5 attendent un préfixe différent pour les requêtes et les passages.
const IS_E5 = /(^|\/)(multilingual-)?e5-/i.test(MODEL_ID);
const QUERY_PREFIX = IS_E5 ? 'query: ' : '';
const PASSAGE_PREFIX = IS_E5 ? 'passage: ' : '';

function maxTokens() {
  if (config.embedMaxTokens > 0) return config.embedMaxTokens;
  const known = KNOWN_LIMITS.find(([re]) => re.test(MODEL_ID));
  return known ? known[1] : 256;
}

let embedderPromise;
function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline('feature-extraction', MODEL_ID, { quantized: true });
  }
  return embedderPromise;
}

let tokenizerPromise;
function getTokenizer() {
  if (!tokenizerPromise) tokenizerPromise = AutoTokenizer.from_pretrained(MODEL_ID);
  return tokenizerPromise;
}

// Nombre de tokens réellement vus par le modèle, préfixe et tokens spéciaux compris.
async function countPassageTokens(text) {
  const tok = await getTokenizer();
  return tok.encode(PASSAGE_PREFIX + text).length;
}

function toVectors(out, count) {
  if (Array.isArray(out)) return out.map(x => Array.from(x.data));
  const [batch, dim] = out.dims.length === 2 ? out.dims : [count, out.data.length / count];
  const res = [];
  for (let i = 0; i < batch; i++) res.push(Array.from(out.data.slice(i * dim, (i + 1) * dim)));
  return res;
}

async function embedTexts(texts) {
  if (!texts.length) return [];
  const m = await getEmbedder();
  const out = await m(texts, { pooling: 'mean', normalize: true });
  return toVectors(out, texts.length);
}

const embedQueries = (texts) => embedTexts(texts.map(t => QUERY_PREFIX + t));
const embedPassages = (texts) => embedTexts(texts.map(t => PASSAGE_PREFIX + t));

module.exports = { MODEL_ID, maxTokens, countPassageTokens, embedQueries, embedPassages };
