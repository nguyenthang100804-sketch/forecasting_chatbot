import React, { useState } from 'react';
import AgentChatPage from './pages/AgentChatPage';
import { CopilotKit } from '@copilotkit/react-core';
import '@copilotkit/react-ui/styles.css';

function App() {
  const RUNTIME_URL = import.meta.env.VITE_COPILOTKIT_RUNTIME_URL
    || '/api/copilotkit';
  const [threadId] = useState(() => crypto.randomUUID());

  return (
    <CopilotKit agent="default" runtimeUrl={RUNTIME_URL} threadId={threadId}>
      <AgentChatPage />
    </CopilotKit>
  );
}

export default App;
