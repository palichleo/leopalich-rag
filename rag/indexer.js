// rag/indexer.js
// Réindexe tout knowledges/ : la collection est vidée puis reconstruite, ce qui évite les doublons.
const fs = require('fs');
const path = require('path');
const config = require('./config');
const { maxTokens, countPassageTokens } = require('./embedder');

const ROOT = path.resolve(__dirname, '..', 'knowledges');

function extractYear(text) {
  const m = (text || '').match(/\b(19|20)\d{2}\b/);
  return m ? parseInt(m[0], 10) : null;
}

// Markdown d'une ligne -> texte brut (liens gardés comme texte, emphase retirée).
function cleanLine(line) {
  return line
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/^\s*(?:[-*+]|\d+[.)])\s+/, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/__([^_]+)__/g, '$1')
    .replace(/(^|[\s(])[*_]([^*_\s][^*_]*)[*_](?=[\s).,;:!?]|$)/g, '$1$2')
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function splitIntoSentences(text) {
  return text
    .split(/(?<=[.?!…])\s+(?=[A-ZÀ-ÖØ-Ý0-9«"])/)
    .map(s => s.trim())
    .filter(Boolean);
}

// Une puce ou une ligne sans ponctuation finale reçoit un point, pour ne pas se fondre dans la
// phrase suivante une fois les lignes réunies.
function terminate(unit) {
  return /[.!?…:;]$/.test(unit) ? unit : `${unit}.`;
}

// Sections délimitées par les titres, chacune avec son chemin H1 > H2 > H3.
// Le premier H1 d'un fichier est son titre : il reste en tête du chemin même quand le fichier
// contient d'autres H1 (« # Description technique »), sinon ces extraits perdraient le nom du
// projet. Les blocs de code sont ignorés : ils n'apportent rien aux questions posées au RAG.
function splitSections(markdown) {
  const sections = [];
  const hStack = [];
  let docTitle = null;
  let lines = [];
  let inFence = false;

  const close = () => {
    const units = lines.flatMap(l => splitIntoSentences(cleanLine(l))).map(terminate);
    const headings = hStack.filter(Boolean);
    if (docTitle && headings[0] !== docTitle) headings.unshift(docTitle);
    if (units.length) sections.push({ hpath: headings.join(' > '), units });
    lines = [];
  };

  for (const line of markdown.split(/\r?\n/)) {
    if (line.trim().startsWith('```')) { inFence = !inFence; continue; }
    if (inFence) continue;
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      close();
      const level = h[1].length;
      const title = cleanLine(h[2]);
      if (level === 1 && docTitle === null) docTitle = title;
      hStack.length = level - 1;
      hStack[level - 1] = title;
      continue;
    }
    if (line.trim()) lines.push(line);
  }
  close();
  return sections;
}

function withHeader(hpath, year, body) {
  return `${hpath ? `[${hpath}] ` : ''}${year ? `[${year}] ` : ''}${body}`;
}

// Coupe une phrase trop longue en morceaux qui tiennent dans le budget.
async function splitLongUnit(unit, fits) {
  if (await fits([unit])) return [unit];
  const words = unit.split(' ');
  if (words.length < 2) return [unit]; // un seul mot : le modèle tronquera
  const mid = Math.ceil(words.length / 2);
  return [
    ...await splitLongUnit(words.slice(0, mid).join(' '), fits),
    ...await splitLongUnit(words.slice(mid).join(' '), fits),
  ];
}

/**
 * Regroupe les phrases d'une section en extraits dont la longueur, mesurée avec le tokenizer du
 * modèle (en-tête et préfixe compris), ne dépasse pas maxTokens. Chaque nouvel extrait reprend
 * les dernières phrases du précédent, dans la limite de overlapTokens.
 */
async function chunkSection({ hpath, units }, { maxTokens: budget, overlapTokens, countTokens }) {
  const year = extractYear(units.join(' ')) || extractYear(hpath);
  const fits = async (parts) =>
    (await countTokens(withHeader(hpath, year, parts.join(' ')))) <= budget;
  const emptyCost = await countTokens('');

  const pieces = [];
  for (const u of units) pieces.push(...await splitLongUnit(u, fits));

  const chunks = [];
  let current = [];
  let fresh = 0; // phrases ajoutées depuis la dernière reprise

  const flush = () => {
    if (fresh > 0) chunks.push(current.join(' '));
  };

  for (const piece of pieces) {
    if (current.length && !(await fits([...current, piece]))) {
      flush();
      // Reprise : dernières phrases de l'extrait qui vient d'être fermé.
      const carry = [];
      for (let i = current.length - 1; i > 0; i--) {
        const candidate = [current[i], ...carry];
        const size = (await countTokens(candidate.join(' '))) - emptyCost;
        if (size > overlapTokens) break;
        carry.unshift(current[i]);
      }
      current = carry;
      fresh = 0;
      if (current.length && !(await fits([...current, piece]))) current = [];
    }
    current.push(piece);
    fresh++;
  }
  flush();

  return chunks.map(body => ({ hpath, year, text: withHeader(hpath, year, body) }));
}

// options.countTokens permet de tester le découpage sans télécharger le tokenizer.
async function chunkMarkdown(markdown, options = {}) {
  const budget = Math.min(options.maxTokens || config.chunkMaxTokens, maxTokens());
  const overlapTokens = options.overlapTokens ?? config.chunkOverlapTokens;
  const countTokens = options.countTokens || countPassageTokens;
  const chunks = [];
  for (const section of splitSections(markdown)) {
    chunks.push(...await chunkSection(section, { maxTokens: budget, overlapTokens, countTokens }));
  }
  return chunks;
}

function toDocs(file, chunks) {
  const docs = [];
  chunks.forEach((c, idx) => {
    const meta = { source: file, chunk: idx, hpath: c.hpath };
    if (c.year) meta.year = c.year;
    docs.push({ id: `${file}#${idx}`, text: c.text, metadata: meta });

    // Option : chaque phrase devient aussi un document court (meilleur rappel, index plus gros).
    if (config.sentenceExplode) {
      const body = c.text.replace(/^(\[[^\]]*\] )+/, '');
      splitIntoSentences(body).forEach((s, j) => {
        docs.push({
          id: `${file}#${idx}:s${j}`,
          text: withHeader(c.hpath, c.year, s),
          metadata: { ...meta, sentence: j },
        });
      });
    }
  });
  return docs;
}

async function indexDir(dir = ROOT) {
  const { resetCollection, addDocuments } = require('./retriever');
  const files = fs.readdirSync(dir).filter(f => /\.(md|txt)$/i.test(f)).sort();
  await resetCollection();

  let totalChunks = 0, totalDocs = 0;
  for (const file of files) {
    const raw = fs.readFileSync(path.join(dir, file), 'utf8');
    const chunks = await chunkMarkdown(raw);
    const docs = toDocs(file, chunks);
    for (let i = 0; i < docs.length; i += config.indexBatch) {
      await addDocuments(docs.slice(i, i + config.indexBatch));
    }
    totalChunks += chunks.length;
    totalDocs += docs.length;
    console.log(`[${file}] ${chunks.length} extrait(s), ${docs.length} document(s)`);
  }
  console.log(`Indexation terminée : ${files.length} fichier(s), ${totalChunks} extrait(s), ${totalDocs} document(s).`);
}

if (require.main === module) {
  indexDir().catch(e => {
    console.error('Indexation échouée :', e);
    process.exit(1);
  });
}

module.exports = { indexDir, chunkMarkdown, splitSections, splitIntoSentences, toDocs };
