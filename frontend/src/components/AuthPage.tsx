import { useState } from "react";
import { loginUser, registerUser } from "../api/client";

interface AuthPageProps {
  onLoginSuccess: (token: string, email: string) => void;
}

export default function AuthPage({ onLoginSuccess }: AuthPageProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const isLogin = mode === "login";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    if (!email || !password) {
      setError("Email and password are required.");
      return;
    }

    if (!isLogin && password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (!isLogin && password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setLoading(true);
    try {
      if (isLogin) {
        const res = await loginUser(email, password);
        onLoginSuccess(res.access_token, email);
      } else {
        await registerUser(email, password);
        setSuccess("Account created! Logging you in…");
        // Auto-login after register
        const res = await loginUser(email, password);
        setTimeout(() => onLoginSuccess(res.access_token, email), 800);
      }
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? err.message ?? "Something went wrong.";
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  function switchMode() {
    setMode(isLogin ? "register" : "login");
    setError(null);
    setSuccess(null);
    setConfirmPassword("");
  }

  return (
    <div className="min-h-screen w-screen flex items-center justify-center" style={{ background: "var(--color-bg)" }}>
      {/* Decorative background blobs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-accent/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-purple-500/15 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: "1s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-accent/5 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 w-full max-w-md px-4">
        {/* Logo / Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-3">
            <span className="w-4 h-4 rounded-full bg-accent inline-block shadow-lg shadow-accent/40" />
            <span className="text-white text-2xl font-bold tracking-tight" style={{ color: "var(--color-text)" }}>
              NeuralNetworks
            </span>
          </div>
          <p className="text-sm" style={{ color: "var(--color-text-secondary, #9ca3af)" }}>
            {isLogin ? "Sign in to your account" : "Create a new account"}
          </p>
        </div>

        {/* Card */}
        <div className="bg-panel rounded-2xl border border-white/5 shadow-2xl shadow-black/20 overflow-hidden">
          {/* Tab switcher */}
          <div className="flex border-b border-white/5">
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-3.5 text-sm font-medium transition-all ${
                isLogin
                  ? "text-white bg-accent/10 border-b-2 border-accent"
                  : "text-gray-400 hover:text-white"
              }`}
              style={isLogin ? { color: "var(--color-text)" } : {}}
            >
              Sign In
            </button>
            <button
              onClick={() => switchMode()}
              className={`flex-1 py-3.5 text-sm font-medium transition-all ${
                !isLogin
                  ? "text-white bg-accent/10 border-b-2 border-accent"
                  : "text-gray-400 hover:text-white"
              }`}
              style={!isLogin ? { color: "var(--color-text)" } : {}}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-4">
            {/* Email */}
            <div>
              <label htmlFor="auth-email" className="block text-sm font-medium mb-1.5" style={{ color: "var(--color-text-secondary, #9ca3af)" }}>
                Email address
              </label>
              <input
                id="auth-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                className="w-full bg-[#0a0c12] border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/30 transition"
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="auth-password" className="block text-sm font-medium mb-1.5" style={{ color: "var(--color-text-secondary, #9ca3af)" }}>
                Password
              </label>
              <input
                id="auth-password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                autoComplete={isLogin ? "current-password" : "new-password"}
                className="w-full bg-[#0a0c12] border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/30 transition"
              />
            </div>

            {/* Confirm Password (register only) */}
            {!isLogin && (
              <div>
                <label htmlFor="auth-confirm" className="block text-sm font-medium mb-1.5" style={{ color: "var(--color-text-secondary, #9ca3af)" }}>
                  Confirm Password
                </label>
                <input
                  id="auth-confirm"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="••••••••"
                  autoComplete="new-password"
                  className="w-full bg-[#0a0c12] border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent/50 focus:ring-1 focus:ring-accent/30 transition"
                />
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 flex items-center gap-2">
                <span>⚠️</span> {error}
              </div>
            )}

            {/* Success message */}
            {success && (
              <div className="text-sm text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-3 flex items-center gap-2">
                <span>✓</span> {success}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-accent hover:bg-accent/90 text-white font-medium py-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-accent/20 hover:shadow-accent/30 active:scale-[0.98]"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  {isLogin ? "Signing in…" : "Creating account…"}
                </span>
              ) : (
                isLogin ? "Sign In" : "Create Account"
              )}
            </button>

            {/* Mode switch hint */}
            <p className="text-center text-sm" style={{ color: "var(--color-text-muted, #6b7280)" }}>
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button type="button" onClick={switchMode} className="text-accent hover:underline font-medium">
                {isLogin ? "Register" : "Sign In"}
              </button>
            </p>
          </form>
        </div>

        {/* Footer */}
        <p className="text-center text-xs mt-6" style={{ color: "var(--color-text-muted, #6b7280)" }}>
          Neural Network Analyzer • Powered by PyTorch & FastAPI
        </p>
      </div>
    </div>
  );
}
