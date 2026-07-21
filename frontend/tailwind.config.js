/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        panel: "#12141c",
        panelLight: "#1a1d29",
        accent: "#7c5cff",
      },
    },
  },
  plugins: [],
};
