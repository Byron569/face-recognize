import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    // 显式声明测试入口,reconnect 回归与新增 fall 去重测试都不会被静默漏掉
    include: ['tests/**/*.test.mjs', 'src/**/*.test.{ts,tsx}'],
  },
});