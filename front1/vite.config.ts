import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://127.0.0.1:8002',
      '/kb': 'http://127.0.0.1:8002',
      '/auth': 'http://127.0.0.1:8002',
      '/models': 'http://127.0.0.1:8002',
      '/mcp': 'http://127.0.0.1:8002',
      '/static': 'http://127.0.0.1:8002',
      '/upload': 'http://127.0.0.1:8002',
    },
  },
})
