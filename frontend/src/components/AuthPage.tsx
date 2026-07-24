import { useEffect, useState } from "react";
import { useGoogleLogin } from "@react-oauth/google";
import { loginUser, registerUser, googleLogin, githubLogin } from "../api/client";

interface AuthPageProps {
  onLoginSuccess: (token: string, email: string) => void;
}

function getGithubRedirectUri() {
  return `${window.location.origin}${window.location.pathname}`;
}

export default function AuthPage({ onLoginSuccess }: AuthPageProps) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [githubLoading, setGithubLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const isLogin = mode === "login";
  const googleClientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
  const githubClientId = import.meta.env.VITE_GITHUB_CLIENT_ID;
  const isGoogleConfigured = !!googleClientId && googleClientId !== "your-google-client-id-here";
  const isGithubConfigured = !!githubClientId && githubClientId !== "your-github-client-id-here";

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    if (!code || !isGithubConfigured) return;

    window.history.replaceState({}, document.title, window.location.pathname);

    (async () => {
      setError(null);
      setGithubLoading(true);
      try {
        const res = await githubLogin(code, getGithubRedirectUri());
        onLoginSuccess(res.access_token, res.email ?? "GitHub User");
      } catch (err: any) {
        const detail = err?.response?.data?.detail ?? err.message ?? "GitHub sign-in failed.";
        setError(detail);
      } finally {
        setGithubLoading(false);
      }
    })();
  }, [isGithubConfigured, onLoginSuccess]);

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
        onLoginSuccess(res.access_token, res.email ?? email);
      } else {
        await registerUser(email, password);
        setSuccess("Account created! Logging you in…");
        const res = await loginUser(email, password);
        setTimeout(() => onLoginSuccess(res.access_token, res.email ?? email), 800);
      }
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? err.message ?? "Something went wrong.";
      setError(detail);
    } finally {
      setLoading(false);
    }
  }

  const handleGoogleLogin = useGoogleLogin({
    flow: "auth-code",
    onSuccess: async (codeResponse) => {
      setError(null);
      setGoogleLoading(true);
      try {
        const res = await googleLogin(codeResponse.code);
        onLoginSuccess(res.access_token, res.email ?? "Google User");
      } catch (err: any) {
        const detail = err?.response?.data?.detail ?? err.message ?? "Google sign-in failed.";
        setError(detail);
      } finally {
        setGoogleLoading(false);
      }
    },
    onError: () => {
      setError("Google sign-in was cancelled or failed.");
    },
  });

  function onGoogleClick() {
    setError(null);
    if (!isGoogleConfigured) {
      setError("Google OAuth is not configured. Add VITE_GOOGLE_CLIENT_ID to frontend/.env and NNA_GOOGLE_CLIENT_ID to backend/.env.");
      return;
    }
    handleGoogleLogin();
  }

  function handleGithubLogin() {
    setError(null);
    if (!isGithubConfigured) {
      setError("GitHub OAuth is not configured. Add VITE_GITHUB_CLIENT_ID to frontend/.env and NNA_GITHUB_CLIENT_ID to backend/.env.");
      return;
    }
    const redirectUri = getGithubRedirectUri();
    const url = `https://github.com/login/oauth/authorize?client_id=${githubClientId}&redirect_uri=${encodeURIComponent(redirectUri)}&scope=user:email`;
    window.location.href = url;
  }

  function switchMode(nextMode: "login" | "register") {
    setMode(nextMode);
    setError(null);
    setSuccess(null);
    setConfirmPassword("");
  }

  const oauthBusy = googleLoading || githubLoading;

  return (
    <div className="min-h-screen w-screen flex items-center justify-center" style={{ background: "var(--color-bg)" }}>
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-accent/20 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-purple-500/15 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: "1s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-accent/5 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 w-full max-w-md px-4">
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

        <div className="bg-panel rounded-2xl border border-white/5 shadow-2xl shadow-black/20 overflow-hidden">
          <div className="flex border-b border-white/5">
            <button
              type="button"
              onClick={() => switchMode("login")}
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
              type="button"
              onClick={() => switchMode("register")}
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
            <div className="space-y-3">
              <button
                type="button"
                onClick={onGoogleClick}
                disabled={oauthBusy || loading}
                className="w-full flex items-center justify-center gap-3 bg-white hover:bg-gray-50 text-gray-700 font-medium py-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-gray-200 shadow-sm hover:shadow-md active:scale-[0.98]"
              >
                {googleLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4 text-gray-500" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Signing in…
                  </span>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 01-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
                      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 009 18z" fill="#34A853"/>
                      <path d="M3.964 10.71A5.41 5.41 0 013.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.997 8.997 0 000 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
                      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 00.957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
                    </svg>
                    {isLogin ? "Sign in with Google" : "Sign up with Google"}
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={handleGithubLogin}
                disabled={oauthBusy || loading}
                className="w-full flex items-center justify-center gap-3 bg-[#24292f] hover:bg-[#2f363d] text-white font-medium py-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-white/10 shadow-sm hover:shadow-md active:scale-[0.98]"
              >
                {githubLoading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Signing in…
                  </span>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.18.82.63-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.51-1.04 2.18-.82 2.18-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
                    </svg>
                    {isLogin ? "Sign in with GitHub" : "Sign up with GitHub"}
                  </>
                )}
              </button>
            </div>

            <div className="relative flex items-center gap-3">
              <div className="flex-1 h-px bg-white/10" />
              <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--color-text-muted, #6b7280)" }}>
                or
              </span>
              <div className="flex-1 h-px bg-white/10" />
            </div>

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

            {error && (
              <div className="text-sm text-red-300 bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3 flex items-center gap-2">
                <span>⚠️</span> {error}
              </div>
            )}

            {success && (
              <div className="text-sm text-emerald-300 bg-emerald-500/10 border border-emerald-500/20 rounded-xl px-4 py-3 flex items-center gap-2">
                <span>✓</span> {success}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || oauthBusy}
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

            <p className="text-center text-sm" style={{ color: "var(--color-text-muted, #6b7280)" }}>
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <button type="button" onClick={() => switchMode(isLogin ? "register" : "login")} className="text-accent hover:underline font-medium">
                {isLogin ? "Register" : "Sign In"}
              </button>
            </p>
          </form>
        </div>

        <p className="text-center text-xs mt-6" style={{ color: "var(--color-text-muted, #6b7280)" }}>
          Neural Network Analyzer • Powered by PyTorch & FastAPI
        </p>
      </div>
    </div>
  );
}
