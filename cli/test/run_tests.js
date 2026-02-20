const api = require('../src/api');
const cfg = require('../src/config');

async function assert(condition, msg) {
  if (!condition) throw new Error(msg || 'Assertion failed');
}

async function testHealth() {
  console.log('Testing /health...');
  const h = await api.get('/health');
  console.log('health:', h.status);
  await assert(h && h.status, '/health missing status');
}

async function testModels() {
  console.log('Testing /v1/models...');
  const m = await api.get('/v1/models');
  console.log('models count:', (m.data || []).length);
  await assert(Array.isArray(m.data), '/v1/models response malformed');
}

async function testProviders() {
  console.log('Testing /providers...');
  const p = await api.get('/providers');
  console.log('providers keys:', Object.keys(p.providers || {}).length);
  await assert(p.providers, '/providers missing');
}

async function testAuthStorage() {
  console.log('Testing token storage...');
  cfg.setToken('test-token-123');
  const t = cfg.getToken();
  console.log('token stored:', t === 'test-token-123');
  await assert(t === 'test-token-123', 'token did not persist');
  cfg.clearToken();
  await assert(!cfg.getToken(), 'token did not clear');
}

async function testChat() {
  console.log('Testing /v1/chat/completions (stream)...');
  const payload = { messages: [{ role: 'user', content: 'Hello from CLI test' }], stream: true };
  await new Promise((resolve, reject) => {
    let acc = '';
    api.postStream('/v1/chat/completions', payload,
      (chunk) => { acc += chunk; process.stdout.write(chunk); },
      () => { console.log('\n--done--'); resolve(acc); },
      (err) => { reject(err); }
    );
  });
}

async function runAll() {
  try {
    await testHealth();
    await testModels();
    await testProviders();
    await testAuthStorage();
    await testChat();
    console.log('\nAll tests passed');
    process.exit(0);
  } catch (e) {
    console.error('\nTest failed:', e);
    process.exit(2);
  }
}

runAll();
