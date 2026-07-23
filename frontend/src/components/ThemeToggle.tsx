import { useTheme } from "./ThemeContext";

export default function ThemeToggle() {
  const { isDark, toggle } = useTheme();

  return (
    <button
      id="theme-toggle"
      onClick={toggle}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className="
        fixed bottom-6 right-6 z-50
        w-14 h-14 rounded-full
        flex items-center justify-center
        shadow-lg shadow-black/20
        transition-all duration-300 ease-in-out
        hover:scale-110 active:scale-95
        bg-[#e8e0f7] dark:bg-[#2a2440]
        border-2 border-accent/30
        hover:border-accent/60
        cursor-pointer
      "
      title={isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
    >
      <span className="text-2xl transition-transform duration-300" style={{ display: "inline-block", transform: isDark ? "rotate(0deg)" : "rotate(360deg)" }}>
        {isDark ? "🌙" : "☀️"}
      </span>
    </button>
  );
}
