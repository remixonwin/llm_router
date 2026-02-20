import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DIR = path.join(os.homedir(), '.llm-router-cli');
export const FILE = path.join(DIR, 'config.json');

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

export function getToken() {
  if (process.env.LLM_ROUTER_TOKEN) return process.env.LLM_ROUTER_TOKEN;
  const cfg = _read();
  return cfg.token || null;
}

export function setToken(token) {
  const cfg = _read();
  cfg.token = token;
  return _write(cfg);
}

export function clearToken() {
  const cfg = _read();
  delete cfg.token;
  return _write(cfg);
}
