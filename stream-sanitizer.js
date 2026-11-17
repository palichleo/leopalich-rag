// stream-sanitizer.js
// Retire le gras Markdown et les accents graves d'une réponse diffusée en flux, y compris quand
// un marqueur est coupé entre deux fragments : les caractères ambigus en fin de fragment
// attendent le fragment suivant. Les astérisques isolés (notes, multiplications) sont gardés.
function createSanitizer() {
  let pending = '';
  const clean = (s) => s
    .replace(/\*\*/g, '')
    .replace(/`/g, '')
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'");
  return {
    push(fragment) {
      const s = pending + fragment;
      const tail = s.match(/[*`]+$/);
      const cut = tail ? s.length - tail[0].length : s.length;
      pending = s.slice(cut);
      return clean(s.slice(0, cut));
    },
    end() {
      const out = clean(pending);
      pending = '';
      return out;
    },
  };
}

module.exports = { createSanitizer };
