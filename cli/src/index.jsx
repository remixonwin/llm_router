import React, { useState } from 'react';
import { render, Box, Text, useInput } from 'ink';
import { Select } from '@inkjs/ui';

import Models from './components/Models.jsx';
import Providers from './components/Providers.jsx';
import Chat from './components/Chat.jsx';
import * as api from './api.js';
import * as cfg from './config.js';
import TextInput from 'ink-text-input';

function App() {
  const [view, setView] = useState('menu');

  const options = [
    { label: 'Models', value: 'models' },
    { label: 'Providers', value: 'providers' },
    { label: 'Chat (streaming)', value: 'chat' },
    { label: 'Auth: Set Token', value: 'auth_set' },
    { label: 'Auth: Clear Token', value: 'auth_clear' },
    { label: 'Health', value: 'health' },
    { label: 'Exit', value: 'exit' },
  ];

  function handleChange(value) {
    if (value === 'exit') process.exit(0);
    setView(value);
  }

  if (view === 'menu') {
    return React.createElement(Box, { flexDirection: 'column' },
      React.createElement(Text, { color: 'cyan' }, 'Intelligent LLM Router — CLI'),
      React.createElement(Text, { dimColor: true }, 'Choose an action:'),
      React.createElement(Select, { options, onChange: handleChange })
    );
  }

  if (view === 'models') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Models, null)));
  if (view === 'providers') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Providers, null)));
  if (view === 'chat') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Chat, null)));

  if (view === 'health') {
    const Health = () => {
      const [h, setH] = useState(null);
      React.useEffect(() => { api.get('/health').then(setH).catch((e) => setH({ error: String(e) })); }, []);
      if (!h) return React.createElement(Text, null, 'Checking health...');
      return React.createElement(Text, null, `Status: ${h.status} — providers ${h.providers_available.length}/${h.providers_total}`);
    };
    return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Health, null)));
  }

  if (view === 'auth_set') {
    const Auth = () => {
      const [val, setVal] = useState('');
      return React.createElement(
        Box,
        { flexDirection: 'column' },
        React.createElement(Text, null, 'Paste API token (will be saved to ~/.llm-router-cli/config.json):'),
        React.createElement(TextInput, { value: val, onChange: setVal, onSubmit: () => { cfg.setToken(val); process.exit(0); } })
      );
    };
    return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Auth, null)));
  }

  if (view === 'auth_clear') {
    cfg.clearToken();
    return React.createElement(Box, { flexDirection: 'column' }, React.createElement(Text, null, 'Token cleared.'), React.createElement(Text, { dimColor: true }, 'Press ENTER to return.'), React.createElement(BackWrapper, null));
  }

  return React.createElement(Text, null, 'Unknown view');
}

function BackWrapper({ children }) {
  const [back, setBack] = useState(false);
  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) process.exit(0);
    if (key.return) setBack(true);
  });
  if (back) return React.createElement(App, null);
  return React.createElement(Box, { flexDirection: 'column' }, children, React.createElement(Text, { dimColor: true }, '\nPress ENTER to return to menu, ESC to quit'));
}

render(React.createElement(App, null));
