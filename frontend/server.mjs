import express from 'express';
import { HttpAgent } from '@ag-ui/client';
import { CopilotRuntime } from '@copilotkit/runtime/v2';
import { createCopilotEndpointExpress } from '@copilotkit/runtime/v2/express';

const port = Number(process.env.COPILOTKIT_RUNTIME_PORT || 4000);
const agentUrl = process.env.LANGGRAPH_AGENT_URL
  || 'http://127.0.0.1:8000/api/copilotkit';

const runtime = new CopilotRuntime({
  agents: {
    default: new HttpAgent({
      agentId: 'default',
      description: 'Chuyên gia dự báo doanh thu và phân tích tác động SHAP bằng LightGBM.',
      url: agentUrl,
    }),
  },
});

const copilotKitRouter = createCopilotEndpointExpress({
  runtime,
  basePath: '/',
});

const app = express();
app.use('/api/copilotkit', copilotKitRouter);

app.listen(port, () => {
  console.log(`CopilotKit runtime: http://127.0.0.1:${port}/api/copilotkit`);
  console.log(`LangGraph agent: ${agentUrl}`);
});
