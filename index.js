// index.js
const config = require('./rag/config'); // charge .env en premier
const express = require('express');
const cors = require('cors');
const { retrieveRelevant } = require('./rag/retriever');
const { createSanitizer } = require('./stream-sanitizer');

const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

const app = express();
app.set('trust proxy', config.trustProxy); // derrière nginx : IP du client dans X-Forwarded-For

// ---------------- Limites (chaque question coûte un appel Groq)
const perIp = new Map();
let day = new Date().toISOString().slice(0, 10);
let dayCount = 0;

function rateLimit(req, res, next) {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== day) { day = today; dayCount = 0; }
  if (dayCount >= config.dailyLimit) {
    return res.status(503).send('Limite quotidienne atteinte, réessayez demain.');
  }

  const now = Date.now();
  const entry = perIp.get(req.ip);
  if (!entry || now > entry.reset) {
    perIp.set(req.ip, { count: 1, reset: now + 60_000 });
  } else if (entry.count >= config.ratePerMinute) {
    res.setHeader('Retry-After', Math.ceil((entry.reset - now) / 1000));
    return res.status(429).send('Trop de questions, réessayez dans une minute.');
  } else {
    entry.count++;
  }
  dayCount++;
  next();
}

setInterval(() => {
  const now = Date.now();
  for (const [ip, e] of perIp) if (now > e.reset) perIp.delete(ip);
}, 60_000).unref();

function trunc(s, max = 800) {
  if (!s || s.length <= max) return s || '';
  return s.slice(0, max) + '…';
}

app.use(cors({ origin: config.allowedOrigins, methods: ['POST', 'GET', 'OPTIONS'], allowedHeaders: ['Content-Type'] }));
app.use(express.json({ limit: '4kb' }));
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  next();
});

app.get('/health', (req, res) => res.status(200).send('OK'));

async function streamGroq(res, systemPrompt, userPrompt, signal) {
  const r = await fetch(GROQ_URL, {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model: config.groqModel,
      stream: true,
      temperature: 0.0,
      top_p: 1,
      max_tokens: 600,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
    }),
  });
  if (!r.ok || !r.body) {
    const errorText = await r.text().catch(() => '');
    throw new Error(`Groq ${r.status} : ${errorText.slice(0, 500)}`);
  }

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('X-Accel-Buffering', 'no');

  const reader = r.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const sanitizer = createSanitizer();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const l = line.trim();
      if (!l.startsWith('data:')) continue;
      const data = l.slice(5).trim();
      if (!data || data === '[DONE]') continue;
      try {
        const tok = JSON.parse(data).choices?.[0]?.delta?.content;
        if (tok) res.write(sanitizer.push(tok));
      } catch {
        // ligne SSE incomplète ou non JSON : ignorée
      }
    }
  }
  res.end(sanitizer.end());
}

app.post('/ask', rateLimit, async (req, res) => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.groqTimeoutMs);
  // Le visiteur a fermé la page : inutile de payer le reste de la génération.
  let clientGone = false;
  res.on('close', () => {
    if (!res.writableEnded) { clientGone = true; controller.abort(); }
  });

  try {
    if (!process.env.GROQ_API_KEY) {
      console.error('GROQ_API_KEY manquant');
      return res.status(500).send('Service indisponible.');
    }
    const rawPrompt = (req.body?.prompt ?? '').toString().trim();
    if (!rawPrompt) return res.status(400).send('Question requise.');
    if (rawPrompt.length > config.promptMaxChars) {
      return res.status(413).send(`Question trop longue (${config.promptMaxChars} caractères au plus).`);
    }

    const relevant = await retrieveRelevant(rawPrompt, 10, 64);
    const context = relevant.map((c, idx) =>
      `- [${idx + 1}] (${c.source}) ${trunc(c.text.replace(/\s+/g, ' ').trim(), 1600)}`
    ).join('\n');

    const today = new Date().toISOString().slice(0, 10);
    const systemPrompt =
      `Tu es un assistant RAG francophone.\n` +
      `RÈGLES:\n` +
      `- Tu réponds UNIQUEMENT à partir des EXTRAITS fournis.\n` +
      `- Si les EXTRAITS sont insuffisants, réponds exactement : "Je n'ai pas accès à cette information."\n` +
      `- Ne fais AUCUNE supposition. Pas d'invention de dates, chiffres, références.\n` +
      `- Réponds en français, en texte simple sans Markdown, clair et factuel (180 mots au plus). Aujourd'hui : ${today}.`;

    const userPrompt =
      `EXTRAITS (numérotés):\n${context}\n\n` +
      `QUESTION:\n${rawPrompt}\n\n` +
      `INSTRUCTIONS:\n` +
      `1) Ne répondre que si l'information apparaît explicitement dans au moins un EXTRAIT.\n` +
      `2) Sinon : "Je n'ai pas accès à cette information."`;

    await streamGroq(res, systemPrompt, userPrompt, controller.signal);
  } catch (err) {
    if (clientGone) return;
    console.error('ERREUR /ask :', err);
    if (!res.headersSent) {
      const timedOut = controller.signal.aborted;
      res.status(timedOut ? 504 : 500).send('Erreur serveur, réessayez plus tard.');
    } else {
      res.end();
    }
  } finally {
    clearTimeout(timeout);
  }
});

app.listen(config.port, '0.0.0.0', () => {
  console.log(`Serveur prêt sur http://localhost:${config.port} (génération : ${config.groqModel})`);
});
