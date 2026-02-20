const React = require('react');
const { render, Box, Text } = require('ink');
const SelectInput = require('ink-select-input').default;
const { useState } = React;

const Models = require('./components/Models');
const Providers = require('./components/Providers');
const Chat = require('./components/Chat');

function App() {
  const [view, setView] = useState('menu');

  const items = [
    { label: 'Models', value: 'models' },
    { label: 'Providers', value: 'providers' },
    { label: 'Chat (streaming)', value: 'chat' },
    { label: 'Health', value: 'health' },
    { label: 'Exit', value: 'exit' },
  ];

  function handleSelect(item) {
    if (item.value === 'exit') process.exit(0);
    setView(item.value);
  }

  if (view === 'menu') {
    return React.createElement(Box, { flexDirection: 'column' },
      React.createElement(Text, { color: 'cyan' }, 'Intelligent LLM Router — CLI'),
      React.createElement(Text, { dimColor: true }, 'Choose an action:'),
      React.createElement(SelectInput, { items, onSelect: handleSelect })
    );
  }

  if (view === 'models') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Models, null)));
  if (view === 'providers') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Providers, null)));
  if (view === 'chat') return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Chat, null)));

  if (view === 'health') {
    const api = require('./api');
    const { useEffect, useState } = React;
    const Health = () => {
      const [h, setH] = useState(null);
      useEffect(() => { api.get('/health').then(setH).catch((e) => setH({ error: String(e) })); }, []);
      if (!h) return React.createElement(Text, null, 'Checking health...');
      return React.createElement(Text, null, `Status: ${h.status} — providers ${h.providers_available.length}/${h.providers_total}`);
    };
    return React.createElement(Box, { flexDirection: 'column' }, React.createElement(BackWrapper, null, React.createElement(Health, null)));
  }

  return React.createElement(Text, null, 'Unknown view');
}

function BackWrapper({ children }) {
  const { useInput } = require('ink');
  const [back, setBack] = useState(false);
  useInput((input, key) => {
    if (key.escape || (key.ctrl && input === 'c')) process.exit(0);
    if (key.return) setBack(true);
  });
  if (back) return React.createElement(App, null);
  return React.createElement(Box, { flexDirection: 'column' }, children, React.createElement(Text, { dimColor: true }, '\nPress ENTER to return to menu, ESC to quit'));
}

render(React.createElement(App, null));
