const fs = require('fs');
const path = require('path');
const os = require('os');

const DIR = path.join(os.homedir(), '.llm-router-cli');
const FILE = path.join(DIR, 'config.json');

function _ensureDir() {
  try {
    fs.mkdirSync(DIR, { recursive: true, mode: 0o700 });
  } catch (e) {
    // ignore
  }
}

function _read() {
  try {
    if (!fs.existsSync(FILE)) return {};
    const raw = fs.readFileSync(FILE, 'utf8');
    return JSON.parse(raw || '{}');
  } catch (e) {
    return {};
  }
}

function _write(obj) {
  _ensureDir();
  try {
    fs.writeFileSync(FILE, JSON.stringify(obj, null, 2), { mode: 0o600 });
    return true;
  } catch (e) {
    return false;
  }
}

function getToken() {
  if (process.env.LLM_ROUTER_TOKEN) return process.env.LLM_ROUTER_TOKEN;
  const cfg = _read();
  return cfg.token || null;
}

function setToken(token) {
  const cfg = _read();
  cfg.token = token;
  return _write(cfg);
}

module.exports = { getToken, setToken, FILE };
