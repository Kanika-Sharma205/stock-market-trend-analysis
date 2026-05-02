/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy:  { 900: '#0f172a', 800: '#1e293b', 700: '#334155', 600: '#475569' },
        brand: { blue: '#38bdf8', indigo: '#818cf8', green: '#34d399', red: '#f87171' },
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
