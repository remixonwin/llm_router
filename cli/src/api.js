const { request } = require('undici');

const BASE = process.env.LLM_ROUTER_URL || 'http://localhost:8080';

async function get(path) {
  const url = `${BASE}${path}`;
  const { body } = await request(url, { method: 'GET' });
  const text = await body.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return text;
  }
}

async function post(path, data) {
  const url = `${BASE}${path}`;
  const { body } = await request(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(data),
  });
  const text = await body.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return text;
  }
}

// Post with streaming support. onChunk will be called with string chunks as they
// arrive. Returns a Promise that resolves when stream ends.
async function postStream(path, data, onChunk, onDone, onError) {
  const url = `${BASE}${path}`;
  try {
    const { body } = await request(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(data),
    });

    // Node stream readable
    if (body && typeof body.on === 'function') {
      body.setEncoding && body.setEncoding('utf8');
      body.on('data', (chunk) => {
        try {
          onChunk(typeof chunk === 'string' ? chunk : chunk.toString('utf8'));
        } catch (e) {
          // swallow handler errors
        }
      });
      body.on('end', () => onDone && onDone());
      body.on('error', (err) => onError && onError(err));
      return;
    }

    // Fallback: attempt to read as text
    const text = await body.text();
    onChunk && onChunk(text);
    onDone && onDone();
  } catch (err) {
    onError && onError(err);
  }
}

module.exports = { get, post, postStream };
