import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/kb': 'http://127.0.0.1:8000',
      '/auth': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/mcp': 'http://127.0.0.1:8000',
      '/static': 'http://127.0.0.1:8000',
      '/upload': 'http://127.0.0.1:8000',
    },
  },
})
