/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Clean monochrome base
        surface: {
          DEFAULT: '#0a0a0a',
          raised: '#141414',
          overlay: '#1a1a1a',
          border: '#2a2a2a',
        },
        // Subtle accent - warm grey with slight teal
        accent: {
          DEFAULT: '#3d9991',
          muted: '#2a6b66',
          subtle: '#1a4542',
        },
        // Semantic colors - muted, professional
        status: {
          success: '#4ade80',
          warning: '#fbbf24',
          error: '#f87171',
          info: '#60a5fa',
        },
      },
    },
  },
  plugins: [],
};
