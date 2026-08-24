import '@testing-library/jest-dom/vitest';

// antd Grid/响应式依赖 window.matchMedia,jsdom 未实现,补一个最小可用实现。
if (typeof window !== 'undefined' && !window.matchMedia) {
  window.matchMedia = ((query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as unknown as MediaQueryList);
}