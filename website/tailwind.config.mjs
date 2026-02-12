/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        quantum: {
          50: '#edf4ff',
          100: '#d8e8ff',
          200: '#b4d2ff',
          300: '#84b5ff',
          400: '#4d91fb',
          500: '#296de8',
          600: '#1f56bf',
          700: '#194491',
          800: '#16396f',
          900: '#152f58',
          950: '#0c1a35',
        },
        neon: {
          cyan: '#4fd8ff',
          green: '#28f0a4',
          copper: '#c38753',
          purple: '#a855f7',
          pink: '#f472b6',
        },
        surface: {
          950: '#040812',
          900: '#07101f',
          850: '#0b1629',
          800: '#0e1d34',
          700: '#1a2b46',
          600: '#2b3d5a',
          500: '#4a5f83',
          400: '#6a7c9f',
          300: '#9fadc5',
        },
      },
      fontFamily: {
        sans: ['Space Grotesk', 'Sora', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['IBM Plex Mono', 'JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'gradient': 'gradient 8s ease infinite',
        'slide-up': 'slideUp 0.5s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'fade-in': 'fadeIn 0.5s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(41, 109, 232, 0.25)' },
          '100%': { boxShadow: '0 0 40px rgba(79, 216, 255, 0.35)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'grid-pattern': 'linear-gradient(rgba(79, 216, 255, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(79, 216, 255, 0.08) 1px, transparent 1px)',
      },
      backgroundSize: {
        'grid': '60px 60px',
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
          },
        },
      },
    },
  },
  plugins: [],
};
