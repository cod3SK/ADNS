/**
 * Vitest config for frontend unit tests in tests/frontend/.
 * Run from repo root:  npx vitest run  (uses node_modules from frontend/)
 */
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { fileURLToPath } from 'url'

const ROOT = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Let test imports of '../../frontend/.../App.jsx' resolve normally
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['tests/frontend/**/*.test.{js,jsx}'],
    setupFiles: ['tests/frontend/setup.js'],
  },
})
