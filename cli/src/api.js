const { request } = require('undici');

const BASE = process.env.LLM_ROUTER_URL || 'http://localhost:8080';
const config = require('./config');

function _authHeaders() {
  const token = config.getToken();
  const headers = { 'content-type': 'application/json' };
  if (token) headers['authorization'] = `Bearer ${token}`;
  return headers;
}

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
    headers: _authHeaders(),
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
      headers: _authHeaders(),
      body: JSON.stringify(data),
    });

    // Node stream readable
    if (body && typeof body.on === 'function') {
      body.setEncoding && body.setEncoding('utf8');

      // Robust parser: handle plain chunks, server-sent-events (lines starting
      // with "data: "), and partial JSON fragments. We accumulate a buffer
      // and emit cleaned messages via onChunk.
      let buf = '';

      function flushLine(line) {
        if (!line) return;
        // SSE style: "data: {...}" or multiple data: lines; strip prefix
        if (line.startsWith('data:')) {
          const v = line.replace(/^data:\s*/i, '');
          try {
            const parsed = JSON.parse(v);
            onChunk(JSON.stringify(parsed));
            return;
          } catch (e) {
            onChunk(v);
            return;
          }
        }

        // Try to parse as JSON whole message
        try {
          const parsed = JSON.parse(line);
          onChunk(JSON.stringify(parsed));
          return;
        } catch (_) {
          // Not JSON, emit raw
          onChunk(line);
        }
      }

      body.on('data', (chunk) => {
        try {
          const s = typeof chunk === 'string' ? chunk : chunk.toString('utf8');
          buf += s;

          // Split on newlines; keep partial last line in buffer
          const parts = buf.split(/\r?\n/);
          buf = parts.pop() || '';
          for (const part of parts) {
            // Ignore keep-alive lines
            const trimmed = part.trim();
            if (!trimmed) continue;
            flushLine(trimmed);
          }
        } catch (e) {
          // swallow handler errors
        }
      });
      body.on('end', () => {
        if (buf && buf.trim()) flushLine(buf.trim());
        onDone && onDone();
      });
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
