// Tests sans réseau ni ChromaDB : le tokenizer est remplacé par un compte de mots.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const { chunkMarkdown, splitSections, toDocs } = require('../rag/indexer');
const { expandQueriesLocal, temporalBonus, coverage } = require('../rag/retriever');
const { createSanitizer } = require('../stream-sanitizer');

const words = async (s) => s.split(/\s+/).filter(Boolean).length + 2; // + 2 tokens spéciaux

test('les sections gardent le chemin de titres et ignorent le code', () => {
  const md = '# Projet [2025]\n## Objectif\n- **But** : détecter\n```py\nprint(1)\n```\n## Résultats\nTexte.';
  const sections = splitSections(md);
  assert.deepEqual(sections.map(s => s.hpath), ['Projet [2025] > Objectif', 'Projet [2025] > Résultats']);
  assert.deepEqual(sections[0].units, ['But : détecter.']);
});

test('le titre du fichier reste en tête des sections qui suivent un autre H1', () => {
  const md = '# Projet X\n## Identité\nA.\n# Description technique\n## Pile\nB.';
  assert.deepEqual(splitSections(md).map(s => s.hpath), [
    'Projet X > Identité',
    'Projet X > Description technique > Pile',
  ]);
});

test('une puce sans ponctuation ne se colle pas à la phrase suivante', () => {
  const [section] = splitSections('# T\n1. Ingestion\n   - Lecture des fichiers');
  assert.deepEqual(section.units, ['Ingestion.', 'Lecture des fichiers.']);
});

test('un titre H2 sans H1 ne produit pas de chemin vide', () => {
  assert.equal(splitSections('## Seul\nTexte.')[0].hpath, 'Seul');
});

test('aucun extrait ne dépasse le budget, en-tête compris', async () => {
  const phrases = Array.from({ length: 40 }, (_, i) => `Phrase numéro ${i} avec quelques mots de plus.`);
  const md = `# Titre\n${phrases.join(' ')}`;
  const chunks = await chunkMarkdown(md, { maxTokens: 40, overlapTokens: 10, countTokens: words });
  assert.ok(chunks.length > 1);
  for (const c of chunks) assert.ok((await words(c.text)) <= 40, c.text);
});

test('le chevauchement reprend la fin de l’extrait précédent, sans dupliquer la nouvelle phrase', async () => {
  const phrases = Array.from({ length: 12 }, (_, i) => `Phrase ${i} ici.`);
  const chunks = await chunkMarkdown(`# T\n${phrases.join(' ')}`, { maxTokens: 20, overlapTokens: 4, countTokens: words });
  for (let k = 1; k < chunks.length; k++) {
    const prev = chunks[k - 1].text.replace('[T] ', '');
    const cur = chunks[k].text.replace('[T] ', '');
    const lastPrev = prev.match(/Phrase \d+ ici\.$/)[0];
    assert.ok(cur.startsWith(lastPrev), `extrait ${k} : ${cur}`);
    // chaque phrase apparaît au plus une fois par extrait
    const seen = cur.match(/Phrase \d+/g);
    assert.equal(new Set(seen).size, seen.length);
  }
  // toutes les phrases sont indexées
  const all = chunks.map(c => c.text).join(' ');
  for (let i = 0; i < 12; i++) assert.ok(all.includes(`Phrase ${i} ici.`));
});

test('une phrase trop longue est coupée', async () => {
  const long = Array.from({ length: 100 }, (_, i) => `mot${i}`).join(' ');
  const chunks = await chunkMarkdown(`# T\n${long}`, { maxTokens: 30, overlapTokens: 0, countTokens: words });
  assert.ok(chunks.length >= 4);
  for (const c of chunks) assert.ok((await words(c.text)) <= 30);
});

test('identifiants stables et pas de doublon de phrases', async () => {
  const chunks = await chunkMarkdown('# T\nUne phrase. Deux phrases.', { maxTokens: 50, countTokens: words });
  const a = toDocs('f.md', chunks).map(d => d.id);
  const b = toDocs('f.md', chunks).map(d => d.id);
  assert.deepEqual(a, b);
  assert.equal(new Set(a).size, a.length);
});

test('les fichiers de knowledges/ se découpent sans erreur', async () => {
  const dir = path.join(__dirname, '..', 'knowledges');
  for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.md'))) {
    const chunks = await chunkMarkdown(fs.readFileSync(path.join(dir, f), 'utf8'), { maxTokens: 120, countTokens: words });
    assert.ok(chunks.length > 0, f);
    const ids = toDocs(f, chunks).map(d => d.id);
    assert.equal(new Set(ids).size, ids.length, f);
  }
});

test('le bonus temporel ne joue que si la question parle de temps', () => {
  assert.equal(temporalBonus('Quels projets en vision ?', 2026, 2026), 0);
  assert.equal(temporalBonus('Que faisait-il en 2023 ?', 2023, 2026), 0.1);
  assert.equal(temporalBonus('Que faisait-il en 2023 ?', 2025, 2026), 0);
  assert.ok(temporalBonus('Son projet le plus récent ?', 2026, 2026) > temporalBonus('Son projet le plus récent ?', 2023, 2026));
  assert.ok(temporalBonus('Son projet le plus récent ?', 2026, 2026) <= 0.05);
});

test('les variantes de la question restent pertinentes', () => {
  const v = expandQueriesLocal('Quels sont ses projets en IA ?', 3);
  assert.equal(v[0], 'quels sont ses projets en ia');
  assert.ok(v.length <= 4);
  // plus d'inversion de mots
  assert.ok(!v.includes('sont quels'));
  // remplacement sur mot entier : « ia » ne touche pas « media »
  assert.ok(!expandQueriesLocal('media social', 5).some(q => q.includes('medmachine')));
});

test('la couverture ne pénalise pas un extrait long', () => {
  assert.equal(coverage(['yolo', 'detection'], ['yolo', 'detection', 'jeu', 'video', 'temps']), 1);
});

test('le nettoyage du flux gère les marqueurs coupés entre fragments', () => {
  const s = createSanitizer();
  const out = ['Le **mo', 'dèle*', '* utilise `YO', 'LO`. Note* : 2*3.'].map(f => s.push(f)).join('') + s.end();
  assert.equal(out, 'Le modèle utilise YOLO. Note* : 2*3.');
});

test('le nettoyage garde les retours à la ligne', () => {
  const s = createSanitizer();
  assert.equal(s.push('Ligne 1.\n\nLigne 2.') + s.end(), 'Ligne 1.\n\nLigne 2.');
});
