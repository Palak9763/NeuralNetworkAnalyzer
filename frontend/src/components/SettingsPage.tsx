import { useState, useEffect } from "react";
import { getProfile, updateProfile, changePassword, deleteAccount } from "../api/client";
import type { ProfileData } from "../api/client";

interface SettingsPageProps {
  onLogout?: () => void;
}

function SectionHeader({ icon, title }: { icon: string; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-lg">{icon}</span>
      <h3 className="text-white font-semibold text-sm uppercase tracking-wider">{title}</h3>
    </div>
  );
}



function StatusBadge({ type, message }: { type: "success" | "error"; message: string }) {
  return (
    <div className={`flex items-center gap-2 text-xs px-3 py-2 rounded-lg ${
      type === "success" ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
    }`}>
      <span>{type === "success" ? "✓" : "✗"}</span>
      <span>{message}</span>
    </div>
  );
}

export default function SettingsPage({ onLogout }: SettingsPageProps) {

  // Profile state
  const [profile, setProfile] = useState<ProfileData | null>(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [editEmail, setEditEmail] = useState("");
  const [emailSaving, setEmailSaving] = useState(false);
  const [emailStatus, setEmailStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Password change state
  const [currentPwd, setCurrentPwd] = useState("");
  const [newPwd, setNewPwd] = useState("");
  const [confirmPwd, setConfirmPwd] = useState("");
  const [pwdSaving, setPwdSaving] = useState(false);
  const [pwdStatus, setPwdStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);

  // Delete state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");

  useEffect(() => {
    getProfile()
      .then((p) => {
        setProfile(p);
        setEditEmail(p.email);
      })
      .catch(() => setProfile(null))
      .finally(() => setProfileLoading(false));
  }, []);

  const handleEmailUpdate = async () => {
    if (!editEmail || editEmail === profile?.email) return;
    setEmailSaving(true);
    setEmailStatus(null);
    try {
      const updated = await updateProfile({ email: editEmail });
      setProfile(updated);
      localStorage.setItem("nna-email", updated.email);
      setEmailStatus({ type: "success", message: "Email updated successfully" });
    } catch (err: any) {
      setEmailStatus({ type: "error", message: err?.response?.data?.detail || "Failed to update email" });
    } finally {
      setEmailSaving(false);
    }
  };

  const handlePasswordChange = async () => {
    setPwdStatus(null);
    if (newPwd !== confirmPwd) {
      setPwdStatus({ type: "error", message: "New passwords do not match" });
      return;
    }
    if (newPwd.length < 6) {
      setPwdStatus({ type: "error", message: "New password must be at least 6 characters" });
      return;
    }
    setPwdSaving(true);
    try {
      await changePassword(currentPwd, newPwd);
      setPwdStatus({ type: "success", message: "Password changed successfully" });
      setCurrentPwd("");
      setNewPwd("");
      setConfirmPwd("");
    } catch (err: any) {
      setPwdStatus({ type: "error", message: err?.response?.data?.detail || "Failed to change password" });
    } finally {
      setPwdSaving(false);
    }
  };

  const handleDeleteAccount = async () => {
    try {
      await deleteAccount();
      onLogout?.();
    } catch {
      alert("Failed to delete account. Please try again.");
    }
  };

  const providerLabel: Record<string, string> = {
    local: "Email & Password",
    google: "Google",
    github: "GitHub",
  };

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      <div>
        <h2 className="text-white font-bold text-xl">Settings</h2>
        <p className="text-gray-400 text-sm mt-1">Configure the Neural Network Analyzer to your preferences.</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">

        {/* ── Profile ── */}
        <div className="bg-panel rounded-xl p-5 border border-white/5 lg:col-span-2">
          <SectionHeader icon="👤" title="Profile" />
          {profileLoading ? (
            <div className="flex items-center gap-3 py-4">
              <div className="w-5 h-5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
              <span className="text-gray-400 text-sm">Loading profile…</span>
            </div>
          ) : profile ? (
            <div className="grid gap-5 lg:grid-cols-2">
              {/* Profile Info */}
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-full bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center text-xl font-bold text-white uppercase shrink-0">
                    {profile.email.charAt(0)}
                  </div>
                  <div>
                    <div className="text-white font-medium">{profile.email}</div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded-full ${
                        profile.auth_provider === "google"
                          ? "bg-blue-500/20 text-blue-400"
                          : profile.auth_provider === "github"
                          ? "bg-gray-500/20 text-gray-300"
                          : "bg-accent/20 text-accent"
                      }`}>
                        {providerLabel[profile.auth_provider] || profile.auth_provider}
                      </span>
                      {profile.created_at && (
                        <span className="text-xs text-gray-500">
                          Joined {new Date(profile.created_at).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Email edit */}
                <div>
                  <label className="text-sm text-gray-400 block mb-1">Email Address</label>
                  <div className="flex gap-2">
                    <input
                      type="email"
                      value={editEmail}
                      onChange={(e) => setEditEmail(e.target.value)}
                      className="flex-1 bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition"
                    />
                    <button
                      onClick={handleEmailUpdate}
                      disabled={emailSaving || editEmail === profile.email}
                      className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition"
                    >
                      {emailSaving ? "Saving…" : "Update"}
                    </button>
                  </div>
                  {emailStatus && <div className="mt-2"><StatusBadge {...emailStatus} /></div>}
                </div>
              </div>

              {/* Password Change (local accounts only) */}
              <div className="space-y-4">
                {profile.auth_provider === "local" ? (
                  <>
                    <label className="text-sm text-gray-400 block">Change Password</label>
                    <input
                      type="password"
                      placeholder="Current password"
                      value={currentPwd}
                      onChange={(e) => setCurrentPwd(e.target.value)}
                      className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent/50 transition"
                    />
                    <input
                      type="password"
                      placeholder="New password (min 6 characters)"
                      value={newPwd}
                      onChange={(e) => setNewPwd(e.target.value)}
                      className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent/50 transition"
                    />
                    <input
                      type="password"
                      placeholder="Confirm new password"
                      value={confirmPwd}
                      onChange={(e) => setConfirmPwd(e.target.value)}
                      className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-accent/50 transition"
                    />
                    <button
                      onClick={handlePasswordChange}
                      disabled={pwdSaving || !currentPwd || !newPwd || !confirmPwd}
                      className="px-4 py-2 rounded-lg bg-white/5 text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-white/10 transition"
                    >
                      {pwdSaving ? "Changing…" : "Change Password"}
                    </button>
                    {pwdStatus && <StatusBadge {...pwdStatus} />}
                  </>
                ) : (
                  <div className="bg-white/[0.02] rounded-lg p-4 border border-white/5">
                    <div className="text-sm text-gray-400">Password Management</div>
                    <div className="text-xs text-gray-500 mt-1">
                      Your account uses {providerLabel[profile.auth_provider]} sign-in. Password is managed by your OAuth provider.
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-sm">Could not load profile data.</p>
          )}
        </div>



        {/* Export Defaults */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="📤" title="Export Defaults" />
          <div className="space-y-3">
            <div>
              <label className="text-sm text-gray-400 block mb-1">Default Export Format</label>
              <select className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition appearance-none">
                <option value="png">PNG Image</option>
                <option value="svg">SVG Vector</option>
                <option value="json">JSON Data</option>
                <option value="pdf">PDF Report</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-1">Image Resolution</label>
              <select className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition appearance-none">
                <option value="1x">1x (Standard)</option>
                <option value="2x">2x (Retina)</option>
                <option value="4x">4x (Print Quality)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="bg-panel rounded-xl p-5 border border-red-500/20 lg:col-span-2">
          <SectionHeader icon="⚠️" title="Danger Zone" />
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white text-sm font-medium">Delete Account</div>
              <div className="text-gray-500 text-xs mt-0.5">Permanently delete your account and all associated data. This action cannot be undone.</div>
            </div>
            {!showDeleteConfirm ? (
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className="px-4 py-2 rounded-lg border border-red-500/30 text-red-400 text-sm font-medium hover:bg-red-500/10 transition shrink-0"
              >
                Delete Account
              </button>
            ) : (
              <div className="flex items-center gap-2 shrink-0">
                <input
                  type="text"
                  placeholder='Type "DELETE" to confirm'
                  value={deleteConfirmText}
                  onChange={(e) => setDeleteConfirmText(e.target.value)}
                  className="bg-[#0a0c12] border border-red-500/30 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none w-48"
                />
                <button
                  onClick={handleDeleteAccount}
                  disabled={deleteConfirmText !== "DELETE"}
                  className="px-4 py-2 rounded-lg bg-red-600 text-white text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-red-700 transition"
                >
                  Confirm
                </button>
                <button
                  onClick={() => { setShowDeleteConfirm(false); setDeleteConfirmText(""); }}
                  className="px-3 py-2 rounded-lg bg-white/5 text-gray-400 text-sm hover:bg-white/10 transition"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
        </div>

        {/* About */}
        <div className="bg-panel rounded-xl p-5 border border-white/5 lg:col-span-2">
          <SectionHeader icon="ℹ️" title="About" />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "Version", value: "1.0.0" },
              { label: "Framework", value: "React + Vite" },
              { label: "Backend", value: "FastAPI" },
              { label: "Parser", value: "PyTorch + TF + JAX" },
            ].map((item) => (
              <div key={item.label} className="bg-[#0a0c12] rounded-xl p-3 border border-white/5">
                <div className="text-xs text-gray-500 uppercase tracking-widest">{item.label}</div>
                <div className="text-white font-medium text-sm mt-1">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
