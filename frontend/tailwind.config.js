/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "var(--color-panel)",
        panelLight: "var(--color-panel-light)",
        accent: "#7c5cff",
        surface: "var(--color-surface)",
      },
    },
  },
  plugins: [],
};
