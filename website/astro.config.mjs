import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://yazhsab.github.io',
  base: '/qbitel-bridge',
  output: 'static',
  integrations: [
    tailwind(),
    react(),
    sitemap({ filter: (page) => !page.includes('/404') }),
    mdx(),
  ],
  markdown: {
    shikiConfig: {
      theme: 'one-dark-pro',
      wrap: true,
    },
  },
});
