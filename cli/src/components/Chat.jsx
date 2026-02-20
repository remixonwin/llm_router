import React, { useState } from 'react';
import { Box, Text, useApp, useInput } from 'ink';
import TextInput from 'ink-text-input';
import * as api from '../api.js';

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
            // Try to append with new line for readability
            setResponse((prev) => (prev ? prev + '\n' + chunk : chunk));
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
      loading ? React.createElement(Text, null, 'Sending...') : React.createElement(Text, { color: response && response.error ? 'red' : 'green' }, response),
      React.createElement(Text, { dimColor: true }, '\nPress ENTER to return to menu.'),
      React.createElement(BackListener, { onExit: exit })
    );
  }

  return null;
}

// Simple component to wait for Enter then exit the Ink app (return to menu handler)
function BackListener({ onExit }) {
  useInput((input, key) => {
    if (key.return) onExit();
  });
  return React.createElement(Box, null);
}

export default Chat;
