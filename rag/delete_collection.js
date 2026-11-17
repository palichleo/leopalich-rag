// rag/delete_collection.js
// Vide la base vectorielle. Inutile avant une réindexation : indexer.js le fait déjà.
const config = require('./config');
const { ChromaClient } = require('chromadb');

(async () => {
  const chroma = new ChromaClient({ host: config.chromaHost, port: config.chromaPort, ssl: false });
  try {
    await chroma.deleteCollection({ name: config.collectionName });
    console.log(`Collection ${config.collectionName} supprimée.`);
  } catch (err) {
    console.error('Erreur suppression collection :', err.message);
    process.exitCode = 1;
  }
})();
