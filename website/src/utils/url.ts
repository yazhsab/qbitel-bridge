/**
 * Prepend the Astro base path to an internal URL.
 * Works with any `base` value in astro.config.mjs.
 *
 * Usage:
 *   import { url } from '@/utils/url';
 *   <a href={url('/products')}>
 */
const base = import.meta.env.BASE_URL.replace(/\/$/, '');

export function url(path: string): string {
  // External URLs and anchors are returned unchanged
  if (path.startsWith('http') || path.startsWith('#')) return path;
  // Ensure path starts with /
  const normalised = path.startsWith('/') ? path : `/${path}`;
  return `${base}${normalised}`;
}
