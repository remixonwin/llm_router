import React, { useEffect, useState } from 'react';
import { Box, Text } from 'ink';
import * as api from '../api.js';

function Models() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    api.get('/v1/models')
      .then((res) => {
        if (!mounted) return;
        setModels(res.data || []);
      })
      .catch((e) => setError(String(e)));
    return () => (mounted = false);
  }, []);

  if (error) return React.createElement(Box, null, React.createElement(Text, { color: 'red' }, `Error: ${error}`));
  if (!models) return React.createElement(Box, null, React.createElement(Text, null, 'Loading models...'));

  return React.createElement(
    Box,
    { flexDirection: 'column' },
    models.slice(0, 50).map((m) =>
      React.createElement(
        Box,
        { key: m.id, marginBottom: 0 },
        React.createElement(Text, { color: 'green' }, m.id),
        React.createElement(Text, { dimColor: true }, ` — ${m.owned_by} [${(m.capabilities || []).join(', ')}]`)
      )
    )
  );
}

export default Models;
