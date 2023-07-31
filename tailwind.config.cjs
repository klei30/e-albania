// tailwind.config.js
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        'custom-blue': 'rgb(20, 66, 81)',
        'custom-pink': 'rgb(237, 23, 78)', // Added custom-pink color
      },
    },
  },
  variants: {},
  plugins: [],
};
