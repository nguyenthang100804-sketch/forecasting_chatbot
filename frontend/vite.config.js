import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/api/copilotkit': {
        target: process.env.COPILOTKIT_RUNTIME_URL || 'http://127.0.0.1:4000',
        changeOrigin: true,
      },
    },
  },
})
