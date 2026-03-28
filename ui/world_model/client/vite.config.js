import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

const SERVER_HOST = process.env.VITE_SERVER_HOST || 'localhost'
const SERVER_PORT = process.env.VITE_SERVER_PORT || '8001'
const target = `http://${SERVER_HOST}:${SERVER_PORT}`

export default defineConfig({
  plugins: [svelte()],
  server: {
    port: 5173,
    proxy: {
      '/ws': { target, ws: true },
      '/status': { target },
      '/models': { target },
    },
  }
})
