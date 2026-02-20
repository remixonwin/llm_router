const React = require('react');
const { Box, Text } = require('ink');
const { useEffect, useState } = React;
const api = require('../api');

function Providers() {
  const [stats, setStats] = useState(null);
  const [cache, setCache] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    api.get('/providers')
      .then((res) => {
        if (!mounted) return;
        setStats(res.providers || {});
        setCache(res.cache || {});
      })
      .catch((e) => setError(String(e)));
    return () => (mounted = false);
  }, []);

  if (error) return React.createElement(Box, null, React.createElement(Text, { color: 'red' }, `Error: ${error}`));
  if (!stats) return React.createElement(Box, null, React.createElement(Text, null, 'Loading provider stats...'));

  return React.createElement(
    Box,
    { flexDirection: 'column' },
    Object.keys(stats).slice(0, 50).map((k) =>
      React.createElement(
        Box,
        { key: k, justifyContent: 'flex-start' },
        React.createElement(Text, { color: stats[k].available ? 'green' : 'red' }, ` ${k} `),
        React.createElement(Text, { dimColor: true }, ` rpd_used=${stats[k].rpd_used} rpm_used=${stats[k].rpm_used}`)
      )
    ),
    React.createElement(Box, { marginTop: 1 }, React.createElement(Text, { dimColor: true }, `Cache: ${JSON.stringify(cache)}`))
  );
}

module.exports = Providers;
