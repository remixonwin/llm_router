const React = require('react');
const { Box, Text, useApp } = require('ink');
const TextInput = require('ink-text-input').default;
const { useState } = React;
const api = require('../api');

function Chat() {
  const { exit } = useApp();
  const [model, setModel] = useState('');
  const [message, setMessage] = useState('');
  const [step, setStep] = useState('model');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  async function send() {
    setLoading(true);
    setResponse('');
    try {
      const payload = {
        model: model || undefined,
        messages: [{ role: 'user', content: message }],
        stream: true,
      };

      // Use streaming POST; append chunks to response
      await new Promise((resolve, reject) => {
        api.postStream(
          '/v1/chat/completions',
          payload,
          (chunk) => {
            setResponse((prev) => (prev ? prev + chunk : chunk));
          },
          () => resolve(),
          (err) => reject(err)
        );
      });
    } catch (e) {
      setResponse((prev) => `${prev}\n[error] ${String(e)}`);
    } finally {
      setLoading(false);
      setStep('done');
    }
  }

  if (step === 'model') {
    return React.createElement(Box, { flexDirection: 'column' },
      React.createElement(Text, null, 'Enter model id (or leave blank for auto):'),
      React.createElement(TextInput, { value: model, onChange: setModel, onSubmit: () => setStep('message') })
    );
  }

  if (step === 'message') {
    return React.createElement(Box, { flexDirection: 'column' },
      React.createElement(Text, null, 'Enter your message:'),
      React.createElement(TextInput, { value: message, onChange: setMessage, onSubmit: send })
    );
  }

  if (step === 'done') {
    return React.createElement(Box, { flexDirection: 'column' },
      loading ? React.createElement(Text, null, 'Sending...') : React.createElement(Text, { color: response && response.error ? 'red' : 'green' }, JSON.stringify(response, null, 2)),
      React.createElement(Text, { dimColor: true }, '\nPress ENTER to return to menu.'),
      React.createElement(BackListener, { onExit: exit })
    );
  }

  return null;
}

// Simple component to wait for Enter then exit the Ink app (return to menu handler)
function BackListener({ onExit }) {
  const { useInput } = require('ink');
  useInput((input, key) => {
    if (key.return) onExit();
  });
  return React.createElement(Box, null);
}

module.exports = Chat;
